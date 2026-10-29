import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

test('posttraining project ZIP binds exactly the seven frozen parent dependencies', () => {
  const python = String.raw`
import ast, hashlib, json, pathlib, shutil, tempfile
from zipfile import ZipFile, ZIP_DEFLATED

root = pathlib.Path.cwd()
builder_tree = ast.parse((root / 'scripts/build-distribution.py').read_text())
checker_tree = ast.parse((root / 'scripts/check-distribution.py').read_text())

def extract(tree, names, function_name):
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == function_name:
            selected.append(node)
    namespace = {'Path': pathlib.Path, 'hashlib': hashlib, 'json': json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<distribution helper>', 'exec'), namespace)
    return namespace

builder = extract(builder_tree, {
    'PROJECT_EXTRA_FILES', 'POSTTRAINING_RECIPE', 'POSTTRAINING_RECIPE_SHA256',
    'POSTTRAINING_PARENT_SHA256'
}, 'posttraining_parent_files')
checker = extract(checker_tree, {
    'POSTTRAINING_RECIPE', 'POSTTRAINING_RECIPE_SHA256',
    'POSTTRAINING_PARENT_SHA256', 'POSTTRAINING_FOUNDATION_ROOT'
}, 'check_posttraining_parent_bundle')
expected = checker['POSTTRAINING_PARENT_SHA256']
recipe = builder['POSTTRAINING_RECIPE']

def make_root(directory):
    destination = pathlib.Path(directory)
    paths = set(expected) | {recipe}
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return destination

def archive_at(path, source_root, entries=None):
    entries = entries if entries is not None else sorted(expected)
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as archive:
        for relative in entries:
            source = source_root / relative
            archive.writestr(relative, source.read_bytes() if source.is_file() else b'unapproved')

with tempfile.TemporaryDirectory() as temporary:
    base = pathlib.Path(temporary)
    source = make_root(base / 'source')
    resolved = builder['posttraining_parent_files'](source)
    assert set(resolved) == set(expected), 'builder must return exactly the seven pinned parent files'

    good_zip = base / 'good.zip'
    archive_at(good_zip, source)
    with ZipFile(good_zip) as archive:
        checker['check_posttraining_parent_bundle'](archive, source)

    def rejected(entries, mutation=None, source_mutation=None):
        bad_zip = base / 'bad.zip'
        archive_at(bad_zip, source, entries)
        if mutation:
            with ZipFile(bad_zip, 'a', compression=ZIP_DEFLATED) as archive:
                mutation(archive)
        with ZipFile(bad_zip) as archive:
            try:
                checker['check_posttraining_parent_bundle'](archive, source)
            except (AssertionError, KeyError, ValueError):
                return
        raise AssertionError('checker accepted an invalid dependency ZIP')

    rejected(sorted(set(expected) - {next(iter(expected))}))
    rejected(sorted(expected) + ['projects/text-harness/foundations/unapproved.py'])
    duplicate_path = next(iter(expected))
    rejected(sorted(expected), lambda archive: archive.writestr(duplicate_path, b'duplicate'))
    tampered_path = next(iter(expected))
    rejected(sorted(expected), lambda archive: archive.writestr(tampered_path, b'changed archive bytes'))

    source_file = source / duplicate_path
    original = source_file.read_bytes()
    source_file.write_bytes(original + b'changed source bytes')
    try:
        try:
            builder['posttraining_parent_files'](source)
        except ValueError as error:
            assert 'hash mismatch' in str(error)
        else:
            raise AssertionError('builder accepted tampered parent source')
        rejected(sorted(expected))
    finally:
        source_file.write_bytes(original)

    recipe_file = source / recipe
    recipe_bytes = recipe_file.read_bytes()
    recipe_file.write_bytes(recipe_bytes + b' ')
    try:
        try:
            builder['posttraining_parent_files'](source)
        except ValueError as error:
            assert 'frozen bridge recipe' in str(error)
        else:
            raise AssertionError('builder accepted a changed bridge recipe')
    finally:
        recipe_file.write_bytes(recipe_bytes)
`;
  const result = spawnSync('python3', ['-c', python], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || result.stdout);
});
