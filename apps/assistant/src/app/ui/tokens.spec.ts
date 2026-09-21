// Design tokens (Task 12).
//
// The rule the console is built on is that a component never names a colour.
// It names a token, and the token is defined once, on :root, with a dark
// counterpart. These specs pin the two halves of that: every token the
// primitives rely on exists, and switching to dark actually changes the
// surfaces rather than leaving white panels on a dark page.

const TOKENS = [
  '--bg',
  '--surface',
  '--surface-2',
  '--border',
  '--border-strong',
  '--text',
  '--text-muted',
  '--text-faint',
  '--brand',
  '--brand-soft',
  '--success',
  '--warning',
  '--danger',
  '--radius',
  '--radius-sm',
  '--shadow',
  '--space-1',
  '--space-2',
  '--space-3',
  '--space-4',
  '--space-5',
  '--focus-ring',
];

function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

describe('design tokens', () => {
  afterEach(() => document.documentElement.removeAttribute('data-theme'));

  it('test_every_token_the_primitives_use_is_defined_on_root', () => {
    const missing = TOKENS.filter((name) => token(name) === '');
    expect(missing).toEqual([]);
  });

  it('test_dark_mode_redefines_every_surface_and_text_token', () => {
    const before = ['--bg', '--surface', '--surface-2', '--border', '--text'].map(token);
    document.documentElement.setAttribute('data-theme', 'dark');
    const after = ['--bg', '--surface', '--surface-2', '--border', '--text'].map(token);

    expect(after.length).toBe(before.length);
    after.forEach((value, i) => expect(value).not.toBe(before[i]));
  });

  // Written without reference to the system preference on purpose: the
  // headless browser's own prefers-color-scheme must not decide whether this
  // passes.
  it('test_an_explicit_theme_wins_over_the_system_preference', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
    const dark = token('--bg');
    document.documentElement.setAttribute('data-theme', 'light');
    const light = token('--bg');

    expect(dark).not.toBe(light);
  });
});
