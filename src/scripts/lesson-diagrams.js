import { $ } from '../lib/dom.js';
import { inlineMath } from '../lib/math.js';

function bindGradientFigure() {
  const px = (x) => 40 + (x + 4) * 50,
    py = (y) => 240 - y * 12;
  $('#gradient-curve').setAttribute(
    'd',
    Array.from({ length: 81 }, (_, i) => {
      const x = -4 + i / 10;
      return `${i ? 'L' : 'M'}${px(x)} ${py(x * x)}`;
    }).join(' '),
  );
  const draw = () => {
    const x = Number($('#gradient-x').value),
      y = x * x,
      slope = 2 * x;
    $('#gradient-point').setAttribute('cx', px(x));
    $('#gradient-point').setAttribute('cy', py(y));
    $('#gradient-tangent').setAttribute(
      'd',
      `M${px(-4)} ${py(y + slope * (-4 - x))}L${px(4)} ${py(y + slope * (4 - x))}`,
    );
    const interpretation =
      x > 0
        ? 'The tangent rises to the right: a small increase in input increases the output.'
        : x < 0
          ? 'The tangent falls to the right: a small increase in input decreases the output.'
          : 'The tangent is horizontal here. The curve still rises on either side, so zero local slope does not mean the entire function is constant.';
    $('#gradient-values').innerHTML = inlineMath(
      String.raw`At \(x=${x.toFixed(1)}\), the point is \((${x.toFixed(1)},${y.toFixed(2)})\) and the tangent slope is \(${slope.toFixed(1)}\). ${interpretation}`,
    );
  };
  $('#gradient-x').oninput = draw;
  draw();
}

function bindAttentionMaskFigure() {
  const update = () => {
    const mode = $('#attention-mode').value,
      values = [2, 4, 8, Number($('#attention-value').value)],
      outputs = [];
    let body = '';
    for (let query = 0; query < 4; query++) {
      const allowed = values.map(
        (_, key) =>
          mode === 'full' ||
          (key <= query && (mode !== 'packed' || Math.floor(query / 2) === Math.floor(key / 2))),
      );
      const count = allowed.filter(Boolean).length;
      outputs.push(values.reduce((sum, value, key) => sum + (allowed[key] ? value / count : 0), 0));
      body += `<tr><th scope="row">Query ${query}</th>${allowed.map((yes, key) => `<td class="${yes ? 'allowed' : 'blocked'}" aria-label="Key ${key}: ${yes ? 'allowed, weight ' + (1 / count).toFixed(3) : 'blocked'}">${yes ? (1 / count).toFixed(2) : '0'}</td>`).join('')}</tr>`;
    }
    $('#attention-grid').innerHTML =
      `<table><caption>Attention weights; each row sums to one</caption><thead><tr><th scope="col">Query / key</th>${values.map((_, i) => `<th scope="col">Key ${i}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
    const worked =
      mode === 'full'
        ? String.raw`Query \(2\) reads all four values: \((2+4+8+${values[3]})/4=${outputs[2].toFixed(3)}\). Every query can read the last value.`
        : mode === 'packed'
          ? String.raw`Query \(2\) is the first token of the second example, so it reads only itself and returns \(8\). Query \(3\) averages its own example: \((8+${values[3]})/2=${outputs[3].toFixed(3)}\).`
          : String.raw`Query \(2\) reads its prefix: \((2+4+8)/3\approx${outputs[2].toFixed(3)}\). Its output stays fixed when the last value changes, because key \(3\) is blocked. Only the last query can read the last value.`;
    $('#attention-result').innerHTML = inlineMath(
      String.raw`Values: \((${values.join(',')})\). Outputs (rounded): \((${outputs.map((x) => Number(x.toFixed(3))).join(',')})\). ${worked}`,
    );
  };
  $('#attention-mode').onchange = update;
  $('#attention-value').oninput = update;
  update();
}

export { bindGradientFigure, bindAttentionMaskFigure };
