export const startHereItems = [
  { label: 'Introduction', slug: 'index' },
  { label: 'Installation', slug: 'guides/installation' },
  { label: 'Usage', slug: 'guides/usage' },
  { label: 'Models', slug: 'models' },
];

export const guidesItems = [
  { label: 'Best practices', slug: 'guides/best_practices' },
  { label: 'Adding a model', slug: 'guides/adding_models' },
  { label: 'Pre-applied beam', slug: 'pre_applied_beam' },
  { label: 'CO line emission', slug: 'co_lines' },
  { label: 'WebSky models', slug: 'websky' },
];

export const notebooksGroup = {
  label: 'Notebooks',
  items: [{ autogenerate: { directory: 'notebooks' } }],
};

export const contributingItems = [
  { label: 'Contribution guide', slug: 'contributing' },
  { label: 'Methods paper contributions', slug: 'pysm_methods_author_contributions' },
];

export function currentSidebar(apiGroup) {
  return [
    { label: 'Start here', items: startHereItems },
    { label: 'Guides', items: guidesItems },
    notebooksGroup,
    apiGroup,
    { label: 'Contributing', items: contributingItems },
  ];
}
