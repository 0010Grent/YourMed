import type { ReactNode } from 'react'

export type FollowUpChipsProps = {
  questions: string[]
  disabled?: boolean
  onPick: (q: string) => void
  header?: ReactNode
}

function uniqNonEmpty(items: string[]) {
  const seen = new Set<string>()
  const out: string[] = []
  for (const it of items) {
    const v = String(it ?? '').trim()
    if (!v) continue
    if (seen.has(v)) continue
    seen.add(v)
    out.push(v)
  }
  return out
}

export default function FollowUpChips(props: FollowUpChipsProps) {
  const questions = uniqNonEmpty(props.questions).slice(0, 3)
  if (!questions.length) return null

  return (
    <div className="followups">
      <div className="followupsHead">{props.header ?? '追问问题'}</div>
      <div className="followupsRow">
        {questions.map((q) => (
          <button
            key={q}
            className="chip"
            disabled={!!props.disabled}
            onClick={() => props.onPick(q)}
            title={q}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
