import { course } from '../course-data.js';
import { currentRelease } from '../releases.js';
import { siteUrl } from '../urls.js';

export const bookDescription =
  'A free JAX and Flax book with worked math, illustrated Python experiments, projects, and lessons on training, evaluation, deployment and Cloud TPUs.';

// Resolve against Astro's configured base so root previews and Pages agree.
export function seoContext(site) {
  const absolute = (path = '') => new URL(siteUrl(path), site).href;
  const home = absolute();
  const author = {
    '@type': 'Person',
    '@id': home + '#author',
    name: 'Taha Bouhsine',
    url: 'https://www.tahabouhsine.com/',
  };
  const provider = {
    '@type': 'Organization',
    '@id': home + '#project',
    name: 'JAX Pathways',
    url: home,
    sameAs: 'https://github.com/mlnomadpy/jaxpathways',
    founder: { '@id': author['@id'] },
  };
  const website = {
    '@type': 'WebSite',
    '@id': home + '#website',
    name: 'JAX Pathways',
    url: home,
    inLanguage: 'en',
    publisher: { '@id': provider['@id'] },
  };
  return { absolute, home, author, provider, website };
}

export function phaseSchema(phase, site) {
  const { absolute, author, provider } = seoContext(site);
  const url = absolute(`phase-guides/${phase.id}.html`);
  const lessons = phase.lessons.filter((lesson) => lesson.status === 'authored');
  return {
    '@type': 'Course',
    '@id': url + '#course',
    url,
    name: phase.title,
    description: phase.description,
    inLanguage: 'en',
    isAccessibleForFree: true,
    author: { '@id': author['@id'] },
    provider: { '@id': provider['@id'] },
    teaches: phase.checkpoint,
    coursePrerequisites: phase.prerequisitePhaseIds
      .map((id) => course.phases.find((item) => item.id === id)?.title)
      .filter(Boolean),
    dateModified: currentRelease.phases[phase.id]?.date,
    // Google's paywall validator also consumes Course.hasPart. Syllabus entries
    // describe lesson coverage without posing as paywall page fragments.
    syllabusSections: lessons.map((lesson) => ({
      '@type': 'Syllabus',
      '@id': absolute(`lesson-${lesson.id}.html`) + '#syllabus',
      url: absolute(`lesson-${lesson.id}.html`),
      name: lesson.title,
      description: lesson.objective,
    })),
  };
}

export function bookSchema(site) {
  const { absolute, author, provider } = seoContext(site);
  return {
    '@type': 'Book',
    '@id': absolute('book.html') + '#book',
    url: absolute('book.html'),
    name: 'JAX Pathways: illustrated course reader',
    description: bookDescription,
    author: { '@id': author['@id'] },
    publisher: { '@id': provider['@id'] },
    inLanguage: 'en',
    isAccessibleForFree: true,
    bookEdition: currentRelease.version,
    dateModified: currentRelease.date,
    bookFormat: 'https://schema.org/EBook',
    encoding: [
      {
        '@type': 'MediaObject',
        contentUrl: absolute('downloads/jax-foundations.epub'),
        encodingFormat: 'application/epub+zip',
        name: 'Download the EPUB book',
      },
      {
        '@type': 'MediaObject',
        contentUrl: absolute('downloads/jax-foundations.html'),
        encodingFormat: 'text/html',
        name: 'Read the illustrated HTML book',
      },
    ],
    about: ['JAX', 'Flax', 'Machine learning', 'Automatic differentiation', 'Cloud TPU'],
  };
}

export function pageSeo({ title, description, canonical, site, structuredData }) {
  const context = seoContext(site);
  const { absolute, home, author, provider, website } = context;
  const relative = canonical.pathname.slice(new URL(home).pathname.length);
  const phase = course.phases.find((item) => relative === `phase-guides/${item.id}.html`);
  const lessonPhase = course.phases.find((item) =>
    item.lessons.some((lesson) => relative === `lesson-${lesson.id}.html`),
  );
  const route = course.pathways.find((item) => relative === `learning-paths/${item.id}.html`);
  const role = course.roles.find((item) => relative === `career-guides/${item.id}.html`);
  let entity = structuredData;
  if (route || role)
    entity = {
      '@type': 'LearningResource',
      '@id': canonical.href + '#guide',
      url: canonical.href,
      name: (route || role).title,
      description: (route || role).description,
      learningResourceType: route ? 'Learning path' : 'Career learning guide',
      author: { '@id': author['@id'] },
      inLanguage: 'en',
      isAccessibleForFree: true,
    };
  if (phase) entity = phaseSchema(phase, site);
  if (relative === 'book.html') entity = bookSchema(site);
  if (relative === 'course.html')
    entity = {
      '@type': 'ItemList',
      '@id': canonical.href + '#courses',
      name: 'JAX courses and phases',
      numberOfItems: course.phases.length,
      itemListElement: course.phases.map((item, index) => ({
        '@type': 'ListItem',
        position: index + 1,
        item: phaseSchema(item, site),
      })),
    };
  const section =
    relative.startsWith('phase-guides/') || relative.startsWith('lesson-')
      ? { name: 'Course catalog', path: 'course.html' }
      : relative.startsWith('project-') || relative.startsWith('tracks/')
        ? { name: 'Projects', path: 'projects.html' }
        : relative.startsWith('learning-paths/')
          ? { name: 'Learning paths', path: 'pathways.html' }
          : relative.startsWith('career-guides/')
            ? { name: 'Careers', path: 'careers.html' }
            : null;
  const crumbs = relative
    ? [
        { name: 'JAX Pathways', url: home },
        ...(section ? [{ name: section.name, url: absolute(section.path) }] : []),
        { name: title.replace(/\s+[|—–-]\s+JAX Pathways$/, ''), url: canonical.href },
      ]
    : [];
  const breadcrumb = crumbs.length
    ? {
        '@type': 'BreadcrumbList',
        '@id': canonical.href + '#breadcrumbs',
        itemListElement: crumbs.map((crumb, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          name: crumb.name,
          item: crumb.url,
        })),
      }
    : null;
  const webpage = {
    '@type': relative === 'course.html' ? 'CollectionPage' : 'WebPage',
    '@id': canonical.href + '#webpage',
    url: canonical.href,
    name: title,
    description,
    inLanguage: 'en',
    isPartOf: { '@id': website['@id'] },
    author: { '@id': author['@id'] },
    ...(breadcrumb ? { breadcrumb: { '@id': breadcrumb['@id'] } } : {}),
    ...(entity ? { mainEntity: { '@id': entity['@id'] || canonical.href + '#resource' } } : {}),
  };
  // JSON-LD is delivered in initial HTML; no client script is required by crawlers.
  return {
    crumbs,
    graph: {
      '@context': 'https://schema.org',
      '@graph': [
        author,
        provider,
        website,
        webpage,
        ...(breadcrumb ? [breadcrumb] : []),
        ...(lessonPhase ? [phaseSchema(lessonPhase, site)] : []),
        ...(entity ? [{ ...entity, '@id': entity['@id'] || canonical.href + '#resource' }] : []),
      ],
    },
  };
}
