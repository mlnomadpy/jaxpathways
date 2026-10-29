import { spawnSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { preparationFor, selectTests, validateInventory } from './test-suites.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
process.chdir(root);

function run(command, args, capture = false) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: 'utf8',
    stdio: capture ? 'pipe' : 'inherit',
    maxBuffer: 16 * 1024 * 1024,
    shell: process.platform === 'win32' && command === 'npm.cmd',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    if (capture) throw new Error(`${command} ${args.join(' ')} failed: ${result.stderr}`);
    process.exit(result.status ?? 1);
  }
  return result.stdout;
}

try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      suite: { type: 'string' },
      changed: { type: 'boolean' },
      since: { type: 'string' },
      list: { type: 'boolean' },
      'no-build': { type: 'boolean' },
      'test-name-pattern': { type: 'string' },
      'test-concurrency': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    console.log(`Usage: npm test -- [--suite unit|app|content|site|all] [options] [test files...]
  --changed              Select from staged, unstaged and untracked paths
  --since <ref>          Also include committed changes since <ref>'s merge base
  --list                 Print selected files and prerequisites without running/building
  --no-build             Reuse generated content/dist (caller guarantees freshness)
  --test-name-pattern X  Filter test names using Node's test runner
  --test-concurrency N   Limit parallel test files (positive integer)
Explicit test files are relative to the repository; --changed accepts changed source paths too.
Default: unit suite. Changed selection defaults to all suites. Browser/Python suites are separate.`);
    process.exit(0);
  }
  validateInventory(root);
  if (values['test-concurrency'] !== undefined && !/^[1-9]\d*$/.test(values['test-concurrency'])) {
    throw new Error('--test-concurrency must be a positive integer.');
  }
  const changed = values.changed || values.since !== undefined;
  let changedFiles;
  if (changed) {
    if (positionals.length) {
      changedFiles = positionals;
    } else {
      const tracked = run(
        'git',
        [
          'diff',
          '--name-only',
          '-z',
          '--no-renames',
          values.since ? `${values.since}...HEAD` : 'HEAD',
        ],
        true,
      );
      const local = values.since
        ? run('git', ['diff', '--name-only', '-z', '--no-renames', 'HEAD'], true)
        : '';
      const untracked = run('git', ['ls-files', '--others', '--exclude-standard', '-z'], true);
      changedFiles = [...new Set(`${tracked}${local}${untracked}`.split('\0').filter(Boolean))];
    }
  }
  let files = selectTests({
    suite: values.suite ?? (changed || positionals.length ? 'all' : 'unit'),
    changedFiles,
    root,
  });
  if (!changed && positionals.length) {
    const requested = positionals.map((file) => file.replace(/^\.\//, '').replaceAll('\\', '/'));
    for (const file of requested) {
      if (!files.includes(file))
        throw new Error(`Unknown test file or outside selected suite: ${file}`);
    }
    files = [...new Set(requested)].sort();
  }
  const preparation = preparationFor(files);
  console.log(
    `Selected ${files.length} Node test files; preparation: ${values['no-build'] ? 'reuse existing artifacts' : preparation}.`,
  );
  if (values.list) {
    for (const file of files) console.log(file);
    process.exit(0);
  }
  if (!files.length) {
    console.log('No affected Node tests. Browser/Python checks use their own commands.');
    process.exit(0);
  }
  if (!values['no-build'] && preparation !== 'none') {
    // Use the same content pipeline as a normal build, once. Release publication is checked by npm run check.
    run(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'content:build']);
    if (preparation === 'site')
      run(process.execPath, ['node_modules/astro/bin/astro.mjs', 'build']);
  }
  const args = ['--test'];
  for (const name of ['test-name-pattern', 'test-concurrency']) {
    if (values[name] !== undefined) args.push(`--${name}=${values[name]}`);
  }
  run(process.execPath, [...args, ...files]);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
