"""Operator execution engine for fundamental signal construction.

Bridges the symbolic operator definitions in operators.py with actual
pandas computations on Compustat firm-year data and CRSP monthly panels.

The engine operates at two levels:
1. Firm-year level: Compustat variables → computed signal (ratios, growth, etc.)
2. Panel level: Firm-year signal → monthly (dates × permnos) panel via FF timing

Usage:
    engine = SignalEngine(comp_ccm, crsp)
    panel = engine.execute("RATIO(SUB(revt, cogs), at)")
    # Returns (nMonths × nStocks) pd.DataFrame
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from siglab.agent.operators import OPERATOR_CATALOG
from siglab.utils.panel import cross_sectional_rank


@dataclass
class SignalEngine:
    """Execute symbolic signal expressions on real data.

    Parameters
    ----------
    comp : Compustat-CCM merged DataFrame with columns:
           permno, year, form_year, and all accounting variables.
           Must be sorted by (permno, year) with lagged variables pre-computed.
    crsp : CRSP monthly DataFrame with columns: permno, date, month, year.
    sic_panel : Optional (dates × permnos) panel of SIC codes for industry adjustment.
    """
    comp: pd.DataFrame
    crsp: pd.DataFrame
    sic_panel: pd.DataFrame | None = None

    def __post_init__(self):
        """Pre-compute lagged variables and CRSP date mapping."""
        # Ensure comp is sorted
        self.comp = self.comp.sort_values(["permno", "year"]).copy()

        # Build list of available raw variables
        self._raw_vars = set(self.comp.columns) - {
            "permno", "gvkey", "datadate", "year", "fyear", "fyr",
            "form_year", "linkdt", "linkenddt",
        }

        # Pre-compute lagged values for ALL numeric variables
        self._ensure_lags()

        # Build CRSP date-to-form_year mapping
        self._crsp_dates = self.crsp[["permno", "date", "month", "year"]].drop_duplicates(
            subset=["permno", "date"]
        )
        self._crsp_dates = self._crsp_dates.copy()
        self._crsp_dates["form_year"] = self._crsp_dates["year"].where(
            self._crsp_dates["month"] >= 7, self._crsp_dates["year"] - 1
        )

        # Lazily-built 2-digit SIC per comp row (see _sic2_series)
        self._sic2_cache: pd.Series | None = None

    def _ensure_lags(self):
        """Pre-compute 1-year lagged values for all numeric variables.

        Uses year-based merge (via _lag_series) so that data gaps produce NaN
        instead of borrowing the nearest available observation via .shift().
        """
        numeric_cols = self.comp.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            lag_col = f"{col}_lag1"
            if lag_col not in self.comp.columns:
                self.comp[lag_col] = self._lag_series(self.comp[col], 1)

    # ── Core execution entry point ───────────────────────────────────────

    def execute(self, expression: str, winsorize_pct: float = 1.0) -> pd.DataFrame:
        """Execute a symbolic signal expression and return a monthly panel.

        Parameters
        ----------
        expression : Symbolic expression, e.g., "RATIO(SUB(revt, cogs), at)"
        winsorize_pct : Winsorize at this percentile (default 1% each tail)

        Returns
        -------
        (nMonths × nStocks) panel of signal values.
        """
        # Parse and compute at firm-year level
        values = self._eval(expression)

        # Winsorize extremes per cross-section (form_year) to avoid look-ahead bias
        if winsorize_pct > 0:
            grp = self.comp["form_year"]
            lo = values.groupby(grp).transform(lambda s: s.quantile(winsorize_pct / 100))
            hi = values.groupby(grp).transform(lambda s: s.quantile(1 - winsorize_pct / 100))
            values = values.clip(lo, hi)

        # Convert to numpy float64 to ensure downstream compatibility
        # (pandas nullable Float64 causes issues with np.isfinite etc.)
        values = values.astype("float64")

        # Attach to comp and build monthly panel
        col_name = "_signal_"
        comp_with_signal = self.comp[["permno", "form_year"]].copy()
        comp_with_signal[col_name] = values.values

        return self._to_monthly_panel(comp_with_signal, col_name)

    # ── Expression parser (recursive descent) ────────────────────────────

    def _eval(self, expr: str) -> pd.Series:
        """Parse and evaluate a symbolic expression at the firm-year level.

        Returns a pd.Series aligned to self.comp's index.
        """
        expr = expr.strip()

        # Try to parse as function call: FUNC(args...)
        match = re.match(r'^(\w+)\((.+)\)$', expr, re.DOTALL)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            # Special handling: check if this is a simple variable (no parens inside)
            # to avoid confusing "at" with a function call
            if func_name.lower() in self._raw_vars and '(' not in args_str:
                # This is actually VAR(something), not a function
                pass
            else:
                args = self._split_args(args_str)
                return self._dispatch(func_name, args)

        # Try to parse as binary arithmetic: A + B, A - B, A * B, A / B
        # (only at the top level, not inside parentheses)
        for op_char, op_name in [('+', 'ADD'), ('-', 'SUB'), ('*', 'MUL'), ('/', 'RATIO')]:
            parts = self._split_binary(expr, op_char)
            if parts:
                left, right = parts
                a = self._eval(left)
                b = self._eval(right)
                return self._binary_op(op_name, a, b)

        # Try to parse as negative: -VAR
        if expr.startswith('-'):
            inner = self._eval(expr[1:])
            return -inner

        # Try to parse as numeric constant
        try:
            val = float(expr)
            return pd.Series(val, index=self.comp.index)
        except ValueError:
            pass

        # Must be a raw variable name
        var_name = expr.strip().lower()
        if var_name in self.comp.columns:
            return self.comp[var_name].copy()

        raise ValueError(f"Unknown variable or expression: '{expr}'")

    def _split_args(self, args_str: str) -> list[str]:
        """Split comma-separated arguments respecting nested parentheses."""
        args = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    def _split_binary(self, expr: str, op: str) -> tuple[str, str] | None:
        """Try to split expr on a binary operator at the top level (depth=0).
        Returns (left, right) or None."""
        depth = 0
        # Scan right to left for - and + to get left-associativity
        # Scan left to right for * and /
        positions = []
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == op and depth == 0 and i > 0:
                # Avoid matching unary minus at position 0
                positions.append(i)

        if not positions:
            return None

        # For - and +, use the LAST position (left-associative)
        # For * and /, use the FIRST position
        if op in ('+', '-'):
            i = positions[-1]
        else:
            i = positions[0]

        left = expr[:i].strip()
        right = expr[i + 1:].strip()
        if left and right:
            return (left, right)
        return None

    # ── Operator dispatch ────────────────────────────────────────────────

    def _dispatch(self, func_name: str, args: list[str]) -> pd.Series:
        """Dispatch a function call to the appropriate operator."""
        name = func_name.upper()

        # Enforce the same arity contract the lab validator applies, so the
        # executor cannot be reached with parameter values the validator
        # would reject (e.g. an extra trailing argument).
        if name in OPERATOR_CATALOG:
            op = OPERATOR_CATALOG[name]
            min_args = op.arity
            max_args = op.arity + len(op.params)
            if not (min_args <= len(args) <= max_args):
                expected = (
                    f"{min_args}" if max_args == min_args else f"{min_args}-{max_args}"
                )
                raise ValueError(
                    f"{name} expects {expected} arguments, got {len(args)}"
                )

        # Binary operators
        if name == "RATIO":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return self._binary_op("RATIO", a, b)
        elif name == "ADD":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return self._binary_op("ADD", a, b)
        elif name == "SUB":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return self._binary_op("SUB", a, b)
        elif name == "MUL":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return self._binary_op("MUL", a, b)
        elif name == "SUMRANK":
            a = self._eval(args[0])
            b = self._eval(args[1])
            grp = self.comp["form_year"]
            ra = a.groupby(grp).rank(pct=True, na_option="keep")
            rb = b.groupby(grp).rank(pct=True, na_option="keep")
            return ra + rb - 1.0
        elif name == "MIN":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return np.minimum(a, b)
        elif name == "MAX":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return np.maximum(a, b)
        elif name == "COALESCE":
            a = self._eval(args[0])
            b = self._eval(args[1])
            return a.where(a.notna(), b)

        # Unary operators with lag/window parameter
        elif name == "GROWTH":
            lag = self._int_param(args, 1, default=1, min_value=1, op_name="GROWTH")
            x_col = args[0].strip().lower()
            if x_col in self.comp.columns:
                return self._growth(x_col, lag)
            else:
                # Complex expression: evaluate at t and t-lag, compute growth
                x_now = self._eval(args[0])
                return self._growth_expr(args[0], lag)
        elif name == "DELTA":
            lag = self._int_param(args, 1, default=1, min_value=1, op_name="DELTA")
            x_col = args[0].strip().lower()
            if x_col in self.comp.columns:
                return self._delta(x_col, lag)
            else:
                return self._delta_expr(args[0], lag)
        elif name == "LAG":
            d = self._int_param(args, 1, default=1, min_value=1, op_name="LAG")
            x_col = args[0].strip().lower()
            if x_col in self.comp.columns:
                return self._lag(x_col, d)
            else:
                # For complex expressions, we need the lagged version
                # Store current value with permno/year, shift within groups
                return self._lag_expr(args[0], d)
        elif name == "MA":
            x = self._eval(args[0])
            k = self._int_param(args, 1, default=3, min_value=1, op_name="MA")
            return self._moving_avg(args[0], k)
        elif name == "VOL":
            k = self._int_param(args, 1, default=5, min_value=2, op_name="VOL")
            return self._rolling_vol(args[0], k)
        elif name == "TREND":
            k = self._int_param(args, 1, default=5, min_value=3, op_name="TREND")
            return self._trend(args[0], k)
        elif name == "ACCEL":
            x_col = args[0].strip().lower()
            if x_col in self.comp.columns:
                return self._accel(x_col)
            return self._accel_expr(args[0])

        elif name == "TS_MIN":
            k = self._int_param(args, 1, default=5, min_value=1, op_name="TS_MIN")
            return self._ts_window(args[0], k).min(axis=1)
        elif name == "TS_MAX":
            k = self._int_param(args, 1, default=5, min_value=1, op_name="TS_MAX")
            return self._ts_window(args[0], k).max(axis=1)
        elif name == "TS_SUM":
            k = self._int_param(args, 1, default=5, min_value=1, op_name="TS_SUM")
            return self._ts_window(args[0], k).sum(axis=1, min_count=1)
        elif name == "TS_COUNT":
            k = self._int_param(args, 1, default=5, min_value=1, op_name="TS_COUNT")
            return self._ts_window(args[0], k).count(axis=1).astype(float)
        elif name == "TS_RANK":
            k = self._int_param(args, 1, default=5, min_value=2, op_name="TS_RANK")
            return self._ts_rank(args[0], k)

        # Math operators
        elif name == "ABS":
            return self._eval(args[0]).abs()
        elif name == "SIGN":
            return np.sign(self._eval(args[0]))
        elif name == "LOG":
            x = self._eval(args[0])
            return np.log(x.where(x > 0))
        elif name == "INV":
            x = self._eval(args[0])
            return 1.0 / x.where(x.abs() > 1e-10)
        elif name == "NEG":
            return -self._eval(args[0])

        # Cross-sectional operators — computed per form_year cross-section
        elif name == "ZSCORE":
            x = self._eval(args[0])
            grp = self.comp["form_year"]
            mean = x.groupby(grp).transform("mean")
            std = x.groupby(grp).transform("std")
            std = std.where(std > 1e-10)
            return (x - mean) / std
        elif name == "RANK":
            x = self._eval(args[0])
            grp = self.comp["form_year"]
            return x.groupby(grp).rank(pct=True, na_option="keep")
        elif name == "WINSOR":
            x = self._eval(args[0])
            p = self._float_param(
                args, 1, default=1.0, op_name="WINSOR", low=0.0, high=50.0
            )
            grp = self.comp["form_year"]
            lo = x.groupby(grp).transform(lambda s: s.quantile(p / 100))
            hi = x.groupby(grp).transform(lambda s: s.quantile(1 - p / 100))
            return x.clip(lo, hi)
        elif name == "INDICATOR":
            x = self._eval(args[0])
            threshold = self._float_param(args, 1, default=0.0, op_name="INDICATOR")
            return (x > threshold).astype(float).where(x.notna())

        # IND_ADJ: industry adjustment
        elif name == "IND_ADJ":
            x = self._eval(args[0])
            return self._ind_adjust(x)

        elif name == "IND_ZSCORE":
            x = self._eval(args[0])
            return self._ind_zscore(x)

        elif name == "IND_RANK":
            x = self._eval(args[0])
            return self._ind_rank(x)

        else:
            raise ValueError(f"Unknown operator: {func_name}")

    # ── Operator implementations ─────────────────────────────────────────

    def _int_param(
        self, args: list[str], idx: int, default: int, min_value: int, op_name: str
    ) -> int:
        """Parse an optional integer parameter at position `idx`, enforcing
        `value >= min_value`. Mirrors PARAM_RULES in siglab.lab.validation
        so the executor rejects what the lab validator rejects."""
        if len(args) <= idx:
            return default
        raw = args[idx].strip()
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(
                f"{op_name} parameter must be an integer >= {min_value}: got '{raw}'"
            )
        if value < min_value:
            raise ValueError(
                f"{op_name} parameter must be >= {min_value}: got {value}"
            )
        return value

    def _float_param(
        self,
        args: list[str],
        idx: int,
        default: float,
        op_name: str,
        low: float | None = None,
        high: float | None = None,
    ) -> float:
        """Parse an optional float parameter at position `idx`. If `low`
        and `high` are given, enforce the open interval `low < value < high`
        (used by WINSOR's percentile bound). Mirrors PARAM_RULES in
        siglab.lab.validation."""
        if len(args) <= idx:
            return default
        raw = args[idx].strip()
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{op_name} parameter must be numeric: got '{raw}'")
        if low is not None and high is not None and not (low < value < high):
            raise ValueError(
                f"{op_name} parameter must be in ({low}, {high}): got {value}"
            )
        return value

    def _binary_op(self, op: str, a: pd.Series, b: pd.Series) -> pd.Series:
        """Apply a binary operator."""
        if op == "RATIO":
            result = a / b.where(b.abs() > 1e-10)
            return result
        elif op == "ADD":
            return a + b
        elif op == "SUB":
            return a - b
        elif op == "MUL":
            return a * b
        else:
            raise ValueError(f"Unknown binary op: {op}")

    def _growth(self, var: str, lag: int = 1) -> pd.Series:
        """Percent change: (X_t - X_{t-lag}) / |X_{t-lag}|."""
        x_now = self.comp[var]
        x_lag = self._lag(var, lag)
        denom = x_lag.abs()
        denom = denom.where(denom > 1e-10)
        return (x_now - x_lag) / denom

    def _delta(self, var: str, lag: int = 1) -> pd.Series:
        """Simple difference: X_t - X_{t-lag}."""
        return self.comp[var] - self._lag(var, lag)

    def _lag(self, var: str, d: int = 1) -> pd.Series:
        """Lagged value: X_{t-d}, using year-based merge to respect data gaps."""
        lag_col = f"{var}_lag1"
        if d == 1 and lag_col in self.comp.columns:
            return self.comp[lag_col]
        return self._lag_series(self.comp[var], d)

    def _ts_window(self, expr: str, k: int) -> pd.DataFrame:
        """Stack an expression's current value and lags 1..k-1 as columns.

        Column 0 is the current value; lags use year-based matching so data
        gaps produce NaN instead of borrowing adjacent observations.
        """
        x_now = self._eval(expr)
        vals = [x_now]
        for d in range(1, k):
            vals.append(self._lag_series(x_now, d))
        return pd.concat(vals, axis=1)

    def _moving_avg(self, expr: str, k: int = 3) -> pd.Series:
        """Moving average over k years.

        Works for both raw variables and complex expressions.
        """
        return self._ts_window(expr, k).mean(axis=1)

    def _rolling_vol(self, expr: str, k: int = 5) -> pd.Series:
        """Rolling standard deviation over k years.

        Works for both raw variables and complex expressions.
        """
        return self._ts_window(expr, k).std(axis=1, ddof=1)

    def _ts_rank(self, expr: str, k: int = 5) -> pd.Series:
        """Percentile position of the current value in its trailing k-year window.

        Average rank of x_t among the window's non-missing observations
        (including itself), divided by the observation count → (0, 1].
        NaN when x_t is missing or fewer than 2 observations are available.
        """
        Y = self._ts_window(expr, k).to_numpy(dtype="float64", na_value=np.nan)
        cur = Y[:, 0]
        n_valid = np.isfinite(Y).sum(axis=1)
        # NaN comparisons are False, so missing lags never count.
        less = (Y < cur[:, None]).sum(axis=1)
        equal = (Y == cur[:, None]).sum(axis=1)  # includes x_t itself
        rank = less + (equal + 1.0) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = rank / n_valid
        pct = np.where(np.isfinite(cur) & (n_valid >= 2), pct, np.nan)
        return pd.Series(pct, index=self.comp.index)

    def _trend(self, expr: str, k: int = 5) -> pd.Series:
        """Linear trend slope over k years.

        Works for both raw variables and complex expressions.
        Uses vectorized regression: slope = Σ(t_i - t̄)(y_i - ȳ) / Σ(t_i - t̄)²

        Fully vectorized — no row-by-row loops.
        """
        x_now = self._eval(expr)
        vals = [x_now]
        for d in range(1, k):
            vals.append(self._lag_series(x_now, d))

        # Stack into (N, k) array
        Y = np.column_stack([v.values for v in vals])  # columns: current, lag1, ...
        t = np.arange(k - 1, -1, -1, dtype=float)
        t_mean = t.mean()
        t_dev = t - t_mean
        ss_t = np.sum(t_dev ** 2)

        # Count valid observations per row
        valid = np.isfinite(Y)
        n_valid = valid.sum(axis=1)

        # For rows with all k observations (most common), vectorized formula
        all_valid = n_valid == k
        slopes = np.full(len(Y), np.nan)

        if all_valid.any():
            Y_full = Y[all_valid]
            y_mean = Y_full.mean(axis=1, keepdims=True)
            y_dev = Y_full - y_mean
            slopes[all_valid] = (y_dev @ t_dev) / ss_t

        # For rows with ≥3 but <k valid obs, use partial regression
        partial = (n_valid >= 3) & ~all_valid
        if partial.any():
            idxs = np.where(partial)[0]
            for i in idxs:
                row = Y[i]
                mask = np.isfinite(row)
                y = row[mask]
                x = t[mask]
                x_mean = x.mean()
                x_dev = x - x_mean
                ss_x = np.sum(x_dev ** 2)
                if ss_x > 0:
                    slopes[i] = np.sum(x_dev * (y - y.mean())) / ss_x

        return pd.Series(slopes, index=self.comp.index)

    def _growth_expr(self, expr: str, lag: int = 1) -> pd.Series:
        """Growth of a complex expression: (f(t) - f(t-lag)) / |f(t-lag)|.

        For complex expressions like GROWTH(RATIO(SUB(revt, cogs), at), 1),
        we evaluate the expression at time t (using current comp data), then
        compute the lagged version using groupby-shift on the evaluated result.
        """
        x_now = self._eval(expr)
        x_lag = self._lag_series(x_now, lag)
        denom = x_lag.abs()
        denom = denom.where(denom > 1e-10)
        return (x_now - x_lag) / denom

    def _delta_expr(self, expr: str, lag: int = 1) -> pd.Series:
        """Delta of a complex expression: f(t) - f(t-lag).

        Evaluates the expression at time t, then shifts by `lag` years
        within each permno to compute the change.
        """
        x_now = self._eval(expr)
        x_lag = self._lag_series(x_now, lag)
        return x_now - x_lag

    def _lag_expr(self, expr: str, d: int = 1) -> pd.Series:
        """Lagged value of a complex expression: f(t-d).

        Evaluates the expression at time t, then shifts by `d` years
        within each permno.
        """
        x_now = self._eval(expr)
        return self._lag_series(x_now, d)

    def _lag_series(self, values: pd.Series, lag: int) -> pd.Series:
        """Shift a firm-year series by `lag` years within each permno.

        Uses year-based matching instead of row-position shift so that data gaps
        (e.g., a firm with years 2020, 2021, 2023 missing 2022) produce NaN for
        the missing lag rather than silently borrowing the nearest observation.
        """
        # Build a reference table: (permno, year+lag) → value
        # When joined back on (permno, year), each row gets the value from lag years ago.
        ref = pd.DataFrame({
            "permno": self.comp["permno"].values,
            "year":   self.comp["year"].values + lag,
            "_lagged": values.values,
        })
        keys = self.comp[["permno", "year"]].reset_index(drop=True)
        merged = keys.merge(ref, on=["permno", "year"], how="left")
        return pd.Series(merged["_lagged"].values, index=values.index)

    def _accel(self, var: str) -> pd.Series:
        """Acceleration: Δ(ΔX) = (X_t - X_{t-1}) - (X_{t-1} - X_{t-2}).

        Uses _lag_series() (year-based merge) for both lags to respect data gaps.
        """
        return self._accel_series(self.comp[var])

    def _accel_expr(self, expr: str) -> pd.Series:
        """Acceleration of a complex expression."""
        return self._accel_series(self._eval(expr))

    def _accel_series(self, values: pd.Series) -> pd.Series:
        """Acceleration of an evaluated firm-year series using year-based lags."""
        x_t0 = values
        x_t1 = self._lag_series(values, 1)
        x_t2 = self._lag_series(values, 2)
        return (x_t0 - x_t1) - (x_t1 - x_t2)

    def _sic2_series(self) -> pd.Series:
        """Two-digit SIC per comp row: Compustat sich primary, CRSP siccd fallback.

        Codes are zero-padded to 4 digits before truncation so SIC < 1000
        (divisions A/B, e.g. 0100 Agriculture) keeps its leading zero.
        Raises when no SIC source exists at all — industry operators must
        never silently no-op.
        """
        if self._sic2_cache is not None:
            return self._sic2_cache

        has_comp_sic = "sic" in self.comp.columns
        if not has_comp_sic and self.sic_panel is None:
            raise ValueError(
                "industry operators (IND_ADJ/IND_ZSCORE/IND_RANK) require SIC "
                "data: comp has no 'sic' column and no sic_panel was provided"
            )

        if has_comp_sic:
            sic = pd.to_numeric(self.comp["sic"], errors="coerce")
        else:
            sic = pd.Series(np.nan, index=self.comp.index)

        if self.sic_panel is not None:
            june = self.sic_panel.loc[self.sic_panel.index.month == 6]
            long = june.stack().rename("siccd").reset_index()
            long.columns = ["date", "permno", "siccd"]
            long["form_year"] = long["date"].dt.year
            crsp_sic = (
                long.dropna(subset=["siccd"])
                .drop_duplicates(subset=["permno", "form_year"], keep="last")
                .set_index(["permno", "form_year"])["siccd"]
            )
            key = pd.MultiIndex.from_arrays(
                [self.comp["permno"], self.comp["form_year"]]
            )
            fill = pd.Series(crsp_sic.reindex(key).to_numpy(), index=self.comp.index)
            sic = sic.fillna(pd.to_numeric(fill, errors="coerce"))

        sic2 = sic.round().astype("Int64").astype("string").str.zfill(4).str[:2]
        self._sic2_cache = sic2
        return sic2

    def _ind_adjust(self, x: pd.Series) -> pd.Series:
        """Industry-adjusted: X - industry_median(X) at each year.
        Industry = 2-digit SIC (Compustat sich, CRSP siccd fallback)."""
        comp = self.comp.copy()
        comp["_signal"] = x
        comp["_sic2"] = self._sic2_series()
        ind_median = comp.groupby(["year", "_sic2"])["_signal"].transform("median")
        return x - ind_median

    def _ind_zscore(self, x: pd.Series) -> pd.Series:
        """Industry z-score: (X - ind_mean) / ind_std."""
        comp = self.comp.copy()
        comp["_signal"] = x
        comp["_sic2"] = self._sic2_series()
        ind_mean = comp.groupby(["year", "_sic2"])["_signal"].transform("mean")
        ind_std = comp.groupby(["year", "_sic2"])["_signal"].transform("std")
        ind_std = ind_std.where(ind_std > 1e-10)
        return (x - ind_mean) / ind_std

    def _ind_rank(self, x: pd.Series) -> pd.Series:
        """Within-industry percentile rank among (year × 2-digit SIC) peers."""
        comp = self.comp.copy()
        comp["_signal"] = x
        comp["_sic2"] = self._sic2_series()
        return comp.groupby(["year", "_sic2"])["_signal"].rank(
            pct=True, na_option="keep"
        )

    # ── Panel construction ───────────────────────────────────────────────

    def _to_monthly_panel(self, comp_data: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        """Map firm-year signal to monthly panel using FF timing convention.

        Timing: form_year = fiscal_year + 1.
        Signal available July form_year through June form_year+1.
        """
        signal_data = comp_data[["permno", "form_year", signal_col]].dropna(subset=[signal_col])
        signal_data = signal_data.drop_duplicates(subset=["permno", "form_year"], keep="last")

        merged = self._crsp_dates.merge(
            signal_data, on=["permno", "form_year"], how="left"
        )

        panel = merged.pivot_table(
            values=signal_col, index="date", columns="permno", aggfunc="first"
        )
        panel.index = pd.to_datetime(panel.index)
        return panel

    # ── Convenience methods ──────────────────────────────────────────────

    def available_variables(self) -> list[str]:
        """Return sorted list of raw Compustat variables available."""
        return sorted(self._raw_vars & set(self.comp.columns))

    def describe_expression(self, expression: str) -> str:
        """Return a human-readable description of what a signal expression computes."""
        try:
            values = self._eval(expression)
            n_valid = values.notna().sum()
            n_total = len(values)
            return (
                f"Expression: {expression}\n"
                f"  Coverage: {n_valid}/{n_total} firm-years ({n_valid/n_total:.1%})\n"
                f"  Mean: {values.mean():.4f}, Median: {values.median():.4f}\n"
                f"  Std: {values.std():.4f}, Min: {values.min():.4f}, Max: {values.max():.4f}"
            )
        except Exception as e:
            return f"Expression: {expression}\n  Error: {e}"
