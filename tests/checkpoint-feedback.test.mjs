import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { lessonCells, lessonMarkdown, markdownText } from '../scripts/lesson-artifacts.mjs';
import { Window } from 'happy-dom';
import { inlineMath } from '../src/lib/math.js';

const course = JSON.parse(readFileSync('public/curriculum.json', 'utf8'));

test('every authored lesson page retains checkpoint controls, code copy and exact section equations', async () => {
  const window = new Window();
  const annotation = window.document.createElement('span');
  try {
    for (const phase of course.phases) {
      for (const lesson of phase.lessons.filter(item => item.status === 'authored')) {
        const html = readFileSync(`dist/lesson-${lesson.id}.html`, 'utf8');
        assert.match(html, /id="lesson-checkpoint"/, lesson.id);
        assert.match(html, /class="copy-snippet"/, lesson.id);
        // Decode only the TeX annotations; hydrating every full lesson repeats the Astro build.
        const equations = new Set([...html.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(([, tex]) => {
          annotation.innerHTML = tex;
          return annotation.textContent;
        }));
        for (const section of lesson.content.sections || []) {
          if (section.math) assert(equations.has(section.math), `reader omitted equation in ${lesson.id}`);
        }
      }
    }
  } finally {
    await window.happyDOM.abort();
  }
});

test('every option keeps its own reasoning in quizzes, Markdown and notebook reveals', () => {
  for (const phase of course.phases) {
    for (const lesson of phase.lessons.filter(item => item.status === 'authored')) {
      const content = lesson.content;
      assert.equal(content.optionFeedback.length, content.options.length, lesson.id);
      assert.equal(new Set(content.optionFeedback).size, content.options.length, lesson.id);
      const quiz = JSON.parse(readFileSync(lesson.artifacts.quizSource, 'utf8')).questions[0];
      assert.deepEqual(quiz.optionFeedback, content.optionFeedback, lesson.id);
      assert.equal(quiz.optionFeedback[quiz.correct], quiz.explanation);
      const markdown = lessonMarkdown(lesson, phase);
      const reveal = markdown.slice(markdown.indexOf('<details><summary>Answer and explanation'));
      const notebookReveal = lessonCells(lesson).find(([kind, text]) => kind === 'markdown' && text.includes('Compare checkpoint reasoning after your attempt'))?.[1];
      assert(notebookReveal?.startsWith('<details>'), lesson.id);
      for (const feedback of content.optionFeedback) {
        assert(reveal.includes(markdownText(feedback)), lesson.id);
        assert(notebookReveal.includes(markdownText(feedback)), lesson.id);
      }
    }
  }
});

test('standalone book keeps wrong-choice explanations inside a deliberate reveal', async () => {
  const window = new Window({ settings: { enableJavaScriptEvaluation: false, disableCSSFileLoading: true } });
  try {
    window.document.write(readFileSync('dist/downloads/jax-foundations.html', 'utf8'));
    const chapters = [...window.document.querySelectorAll('main article')];
    const lessons = course.phases.flatMap(phase => phase.lessons).filter(item => item.status === 'authored');
    for (const [index, lesson] of lessons.entries()) {
      const details = [...chapters[index].querySelectorAll('details')].find(node => node.querySelector('summary')?.textContent.includes('checkpoint answer'));
      assert(details && !details.open, lesson.id);
      assert(details.textContent.includes('Reason through the other choices'), lesson.id);
      for (let i = 0; i < lesson.content.options.length; i++) {
        if (i !== lesson.content.answer) {
          const expected=window.document.createElement('div');
          expected.innerHTML=inlineMath(lesson.content.optionFeedback[i]);
          assert(details.textContent.includes(expected.textContent), lesson.id);
        }
      }
    }
  } finally {
    await window.happyDOM.abort();
  }
});
