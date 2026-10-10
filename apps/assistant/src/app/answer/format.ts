// Formatting that never invents a number (ADR 0004): null and undefined are "n/a".

export const NA = 'n/a';

export function count(value: number | null | undefined): string {
  return value == null ? NA : value.toLocaleString('en-GB');
}

export function ms(value: number | null | undefined): string {
  if (value == null) {
    return NA;
  }
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function score(value: number | null | undefined, digits = 2): string {
  return value == null ? NA : value.toFixed(digits);
}

export function percent(value: number | null | undefined): string {
  return value == null ? NA : `${Math.round(value * 100)}%`;
}

/** An estimated cost in US dollars, labelled as an estimate by the caller. */
export function usd(value: number | null | undefined): string {
  if (value == null) {
    return NA;
  }
  return value < 0.01 ? `$${value.toFixed(5)}` : `$${value.toFixed(4)}`;
}
