// Lays out the figures of one analytics entry so they can be scanned: who or what a figure is about on one line,
// its values beneath with the numbers picked out, and a side-by-side comparison as a small table.
//
// The figures arrive as lines of text from approved-analytics.ts and estimated-analytics.ts, which follow one
// convention: "Label: value · value" for a row, "A – B" then "Label: x – y" rows for a comparison, and a plain
// sentence for anything else. Nothing is calculated here.
import { Fragment } from "react";

type Row = { label: string | null; parts: string[] };

const NUMBER = /(?<![A-Za-z#])(\d[\d,]*(?:\.\d+)?%?)/g;
const MAX_LABEL = 40;  // a longer run of words before a colon is a sentence, not a label

function row(line: string, longest = MAX_LABEL): Row {
  const at = line.indexOf(": ");
  const label = at > 0 && at <= longest && !line.slice(0, at).includes(" · ") ? line.slice(0, at) : null;
  return { label, parts: (label ? line.slice(at + 2) : line).split(" · ") };
}

/** "A – B" followed only by "Label: x – y" rows is a comparison of two sides. */
function comparison(lines: string[]) {
  const head = lines[0]?.includes(": ") ? null : lines[0]?.split(" – ");
  if (!head || head.length !== 2 || lines.length < 2) return null;
  const rows = lines.slice(1).map(line => { const { label, parts } = row(line, 80); const sides = parts.join(" · ").split(" – "); return label && sides.length === 2 ? { label, sides } : null; });
  return rows.every(Boolean) ? { head, rows: rows as { label: string; sides: string[] }[] } : null;
}

/** Text with its numbers picked out, so the eye lands on the figure and the words stay quiet. */
function Marked({ text }: { text: string }) {
  return <>{text.split(NUMBER).map((piece, index) => index % 2 ? <b key={index} className="fig__n">{piece}</b> : <Fragment key={index}>{piece}</Fragment>)}</>;
}

export function Figures({ lines, label }: { lines: string[]; label: string }) {
  const compared = comparison(lines);
  if (compared) {
    return (
      <div className="table-wrap">
        <table className="table fig__table" aria-label={label}>
          <thead><tr><th scope="col"><span className="sr-only">Figure</span></th>{compared.head.map(side => <th key={side} scope="col" className="num">{side}</th>)}</tr></thead>
          <tbody>
            {compared.rows.map(item => (
              <tr key={item.label}><th scope="row">{item.label}</th>{item.sides.map((side, index) => <td key={index} className="num"><Marked text={side} /></td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  // Lines about the same team or player sit under one label, in the order the label first appears.
  const groups: { name: string | null; lines: string[][] }[] = [];
  lines.forEach(line => {
    const { label: name, parts } = row(line);
    const group = name ? groups.find(item => item.name === name) : undefined;
    if (group) group.lines.push(parts); else groups.push({ name, lines: [parts] });
  });
  const values = (parts: string[], index: number) => <span key={index} className="fig__parts">{parts.map(part => <span key={part}><Marked text={part} /></span>)}</span>;
  return (
    <div className="fig" role="group" aria-label={label}>
      {groups.map((group, index) => group.name
        ? <div key={index} className="fig__row"><span className="fig__label">{group.name}</span>{group.lines.map(values)}</div>
        : <p key={index} className="fig__parts fig__lead">{group.lines[0].map(part => <span key={part}><Marked text={part} /></span>)}</p>)}
    </div>
  );
}
