// Display formatting. A missing value is never shown as zero: it reads "—" or "Unknown".

export const DASH = "—";

type Maybe = number | null | undefined;

// Round on the stored value first (the same result the analysis gets when it prints a figure), then
// let the locale add separators. Locale formatting alone can round a stored 60.65 up to 60.7.
const fixed = (value: number, digits: number) => Number(value.toFixed(digits)).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

/**
 * A number with a fixed count of decimals, so figures that sit together never mix "7" with "60.5".
 * One decimal for reading, two in tables, zero for counts. Each value is rounded on its own and never
 * adjusted to make a set add up. A positive value too small to show reads "<0.1" rather than "0.0".
 */
export function num(value: Maybe, digits = 1): string {
  if (value == null || Number.isNaN(value)) return DASH;
  if (digits > 0 && value > 0 && Number(value.toFixed(digits)) === 0) return `<${fixed(10 ** -digits, digits)}`;
  return fixed(value, digits);
}

/** A percentage to one decimal by default. */
export function pct(value: Maybe, digits = 1): string {
  return value == null || Number.isNaN(value) ? "Unknown" : `${num(value, digits)}%`;
}

/** Seconds to one decimal by default; tables may ask for two. Always a space before the unit. */
export function secs(value: Maybe, digits = 1): string {
  return value == null || Number.isNaN(value) ? "Unknown" : `${num(value, digits)} s`;
}

/** A source-time range such as "3.84–4.88 s". */
export function range(start: number, end: number, digits = 2): string {
  return `${fixed(start, digits)}–${fixed(end, digits)} s`;
}

/** Recorded settings such as a frame rate, shown as stored: up to two decimals, no padding. */
export function loose(value: Maybe, maxDigits = 2): string {
  return value == null || Number.isNaN(value) ? DASH : value.toLocaleString(undefined, { maximumFractionDigits: maxDigits });
}

/** m:ss for axes and durations. */
export function clock(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, "0")}`;
}

/** m:ss.t for the playhead and "watch" buttons. */
export function clockTenths(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const tenths = Math.floor(safe * 10 + 1e-6);
  return `${Math.floor(tenths / 600)}:${Math.floor((tenths % 600) / 10).toString().padStart(2, "0")}.${tenths % 10}`;
}

/** snake_case identifiers from the data, made readable. */
export function words(text: string): string {
  return text.replaceAll("_", " ");
}

export function sentence(text: string): string {
  const spaced = words(text);
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
