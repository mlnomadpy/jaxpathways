#!/usr/bin/env python3
"""Browse and run the generated JAX course from a repository checkout. Stdlib CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def load_course():
    return json.loads((ROOT / 'public/curriculum.json').read_text())

def find_lesson(course, lesson_id):
    for phase in course['phases']:
        for lesson in phase['lessons']:
            if lesson['id'] == lesson_id:
                return phase, lesson
    raise ValueError(f'Unknown lesson: {lesson_id}. Use list to see valid IDs.')

def select_lessons(course, pathway=None):
    phases = course['phases']
    if pathway:
        route = next((p for p in course['pathways'] if p['id'] == pathway), None)
        if not route:
            raise ValueError(f'Unknown pathway: {pathway}')
        by_id = {p['id']: p for p in phases}
        phases = [by_id[id] for id in route['phaseIds']]
    return [(p, l) for p in phases for l in p['lessons']]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='List canonical lessons and their content status')
    listing.add_argument('--pathway')
    listing.add_argument('--available', action='store_true')
    for name in ['show', 'run', 'quiz']:
        sub = commands.add_parser(name)
        sub.add_argument('lesson_id')
    plan = commands.add_parser('plan', help='Print a career plan; redirect to LEARNING.md to keep it')
    plan.add_argument('role_id')
    guide = commands.add_parser('guide', help='Read a modality guide and its current harness availability')
    guide.add_argument('track_id', nargs='?')
    args = parser.parse_args()
    course = load_course()
    try:
        if args.command == 'list':
            for phase, lesson in select_lessons(course, args.pathway):
                if args.available and lesson['status'] != 'authored':
                    continue
                print(f"{phase['number']}  {lesson['id']:20}  {lesson['status']:8}  {lesson['title']}")
            return 0
        if args.command == 'guide':
            tracks = course.get('modalityTracks', [])
            if not args.track_id:
                for track in tracks:
                    status = 'runnable harness: ' + track['harnessProjectId'] if track.get('harnessProjectId') else 'guided plan; harness planned'
                    print(f"{track['id']}: {track['title']} ({status})")
                return 0
            track = next((t for t in tracks if t['id'] == args.track_id), None)
            if track is None:
                raise ValueError(f'Unknown track: {args.track_id}. Use guide to list valid IDs.')
            print((ROOT / 'public/guides' / (track['id'] + '.md')).read_text())
            return 0
        if args.command == 'plan':
            role = next((r for r in course['roles'] if r['id'] == args.role_id), None)
            if not role:
                raise ValueError(f'Unknown role: {args.role_id}')
            print(f"# {role['title']}\n\n{role['description']}\n\n## Route\n")
            print(f"Work example: {role['workExample']}\n\nPreparation: {role['background']}\n")
            for track_id in role.get('modalityTrackIds', []):
                track = next(t for t in course['modalityTracks'] if t['id'] == track_id)
                status = 'runnable harness: ' + track['harnessProjectId'] if track.get('harnessProjectId') else 'connected harness planned'
                print(f"Project guide: python scripts/course.py guide {track_id} ({status})")
            route = next(r for r in course['pathways'] if r['id'] == role['defaultPathwayId'])
            print(f"\nFirst artifact: {route['firstArtifact']}\n\n{route['studyAdvice']}\n")
            for phase, lesson in select_lessons(course, role['defaultPathwayId']):
                print(f"- [ ] {phase['number']} / {lesson['title']} ({lesson['status']}; {lesson['id']})")
            print(f"\n## Portfolio\n\n{role['portfolio']}\n\n## Demonstrate\n\n{role['readiness']}\n\nReading, quiz correctness, executable evidence, and expert review are separate. Planned lessons are not authored labs.")
            print(f"\nProject workspace: projects/{route['capstone']['projectId']}/README.md\nAssessment: {route['capstone']['assessmentDraft']['source']}")
            print('\n## Review questions\n\n' + '\n'.join('- '+q for q in role['reviewQuestions']))
            print('\n## Engineering extensions\n\n' + '\n'.join('- '+id for id in route.get('engineeringLessonIds', [])) + '\nFollow each lesson prerequisite; project: projects/engineering-release/README.md')
            print('\n## Evidence to collect\n\n' + '\n'.join('- '+item for item in route['capstone']['evidence']))
            return 0
        phase, lesson = find_lesson(course, args.lesson_id)
        if lesson['status'] != 'authored':
            raise ValueError(f"{lesson['id']} is a planned brief, not an executable lesson.")
        if args.command == 'show':
            print((ROOT / lesson['artifacts']['source']).read_text())
        elif args.command == 'run':
            script = (ROOT / lesson['artifacts']['scriptSource']).resolve()
            if not script.is_relative_to(ROOT / 'phases'):
                raise ValueError('Lesson script is outside the course source')
            print(f"Command: {sys.executable} {script.relative_to(ROOT)}\nWorking directory: {ROOT}\nBackend: CPU", flush=True)
            result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, timeout=60)
            print(f'Exit code: {result.returncode}. This run does not award a reviewed competency.')
            return result.returncode
        elif args.command == 'quiz':
            quiz = json.loads((ROOT / lesson['artifacts']['quizSource']).read_text())
            correct = 0
            for q in quiz['questions']:
                print(q['question'])
                for i, option in enumerate(q['options']):
                    print(f'  {i+1}. {option}')
                answer = input('Choose a number: ').strip()
                passed = answer == str(q['correct'] + 1)
                correct += int(passed)
                print(('Correct. ' if passed else 'Review this concept. ') + q['explanation'])
            print(f"{correct}/{len(quiz['questions'])} correct. Exercise evidence remains separate.")
            return 0 if correct == len(quiz['questions']) else 1
        return 0
    except (ValueError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 2

if __name__ == '__main__':
    sys.exit(main())
