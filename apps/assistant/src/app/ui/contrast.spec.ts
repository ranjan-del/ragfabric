// Colour contrast of the filled and tinted controls, in both themes.
//
// Found by measuring the dark theme on every page: white text on the lighter
// indigo and red fills (about 3:1) and dark green text on the dark green
// "ready" badge (about 2.9:1). The foreground of each of these is now a token
// that dark mode redefines, and this spec holds the pairs to WCAG AA (4.5:1).

const CASES: Array<[string, string]> = [
  ['btn btn-primary', 'primary button'],
  ['btn btn-danger', 'danger button'],
  ['badge badge-success', 'success badge'],
  ['badge badge-warning', 'warning badge'],
  ['badge badge-danger', 'danger badge'],
  ['badge badge-brand', 'brand badge'],
  ['badge', 'neutral badge'],
  ['alert alert-error', 'error alert'],
  ['alert alert-info', 'info alert'],
  ['alert alert-warning', 'warning alert'],
];

function rgb(value: string): [number, number, number] {
  const match = value.match(/rgba?\(([^)]+)\)/);
  if (match === null) {
    throw new Error(`not a colour: ${value}`);
  }
  const [r, g, b] = (match[1] as string).split(/[ ,/]+/).filter(Boolean).map(Number);
  return [r as number, g as number, b as number];
}

function luminance([r, g, b]: [number, number, number]): number {
  const f = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}

function ratio(el: HTMLElement): number {
  const style = getComputedStyle(el);
  const a = luminance(rgb(style.color));
  const b = luminance(rgb(style.backgroundColor));
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

describe('contrast of filled and tinted controls', () => {
  afterEach(() => document.documentElement.removeAttribute('data-theme'));

  for (const theme of ['light', 'dark']) {
    for (const [classes, label] of CASES) {
      it(`test_${label.replace(/ /g, '_')}_is_at_least_4_5_to_1_in_${theme}`, () => {
        document.documentElement.setAttribute('data-theme', theme);
        const el = document.createElement('span');
        el.className = classes;
        el.textContent = 'Sample';
        document.body.appendChild(el);
        try {
          expect(ratio(el)).toBeGreaterThanOrEqual(4.5);
        } finally {
          el.remove();
        }
      });
    }
  }
});
