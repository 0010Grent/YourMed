import type { AgentQuestion } from '../types/agent'

export type FollowUpPanelProps = {
  askText?: string
  questions?: AgentQuestion[]
  nextQuestions?: string[]
  disabled?: boolean
  onSendAnswer: (answerText: string) => void
  // 仅用于 fallback（没有结构化 questions 时）：预填输入框但不自动发送
  onPrefill?: (templateAnswer: string) => void
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

function suggestionsFromPlaceholder(placeholder?: string): string[] {
  const p = String(placeholder ?? '').trim()
  if (!p) return []

  // 例如：24岁 / 20多 / 不确定
  const cleaned = p.replace(/^例如[:：]\s*/g, '')
  const parts = cleaned
    .split('/')
    .map((s) => s.trim())
    .filter(Boolean)
  return uniqNonEmpty(parts).slice(0, 3)
}

function renderEnum(q: AgentQuestion, disabled: boolean, onSendAnswer: (t: string) => void) {
  const choices = Array.isArray(q.choices) ? uniqNonEmpty(q.choices) : []
  if (!choices.length) return null

  return (
    <div className="followupsRow">
      {choices.map((c) => (
        <button key={c} className="chip" disabled={disabled} onClick={() => onSendAnswer(c)} title={c}>
          {c}
        </button>
      ))}
    </div>
  )
}

function renderNumber(q: AgentQuestion, disabled: boolean, onSendAnswer: (t: string) => void) {
  const r = Array.isArray(q.range) && q.range.length === 2 ? q.range : undefined
  const min = r ? Number(r[0]) : 0
  const max = r ? Number(r[1]) : 10
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return null

  const values: number[] = []
  for (let i = min; i <= max; i += 1) values.push(i)

  return (
    <div className="followupsRow">
      {values.map((v) => (
        <button
          key={v}
          className="chip"
          disabled={disabled}
          onClick={() => onSendAnswer(`${v}/10`)}
          title={`${v}/10`}
        >
          {v}
        </button>
      ))}
    </div>
  )
}

function renderText(q: AgentQuestion, disabled: boolean, onSendAnswer: (t: string) => void) {
  const sugg = suggestionsFromPlaceholder(q.placeholder)
  if (!sugg.length) return null

  // 对 text 类型：点击即发送“答案文本”（不发送问题文本）
  return (
    <div className="followupsRow">
      {sugg.map((s) => (
        <button
          key={s}
          className="chip"
          disabled={disabled}
          onClick={() => onSendAnswer(s)}
          title={s}
        >
          {s}
        </button>
      ))}
    </div>
  )
}

export default function FollowUpPanel(props: FollowUpPanelProps) {
  const qs = Array.isArray(props.questions) ? props.questions.filter((q) => q && q.slot && q.question).slice(0, 3) : []
  const fallback = uniqNonEmpty(props.nextQuestions ?? []).slice(0, 3)

  if (!qs.length && !fallback.length) return null

  return (
    <div className="followups">
      <div className="followupsHead">追问（请直接回答）</div>

      {props.askText ? <div className="placeholder" style={{ marginTop: 6 }}>{props.askText}</div> : null}

      {qs.length ? (
        <div style={{ marginTop: 8 }}>
          {qs.map((q) => (
            <div key={`${q.slot}:${q.question}`} style={{ marginTop: 10 }}>
              <div className="placeholder">{q.question}</div>
              {q.type === 'enum' ? renderEnum(q, !!props.disabled, props.onSendAnswer) : null}
              {q.type === 'number' ? renderNumber(q, !!props.disabled, props.onSendAnswer) : null}
              {q.type !== 'enum' && q.type !== 'number'
                ? renderText(q, !!props.disabled, props.onSendAnswer)
                : null}
            </div>
          ))}
        </div>
      ) : (
        <div style={{ marginTop: 8 }}>
          {fallback.map((q) => (
            <div key={q} style={{ marginTop: 10 }}>
              <div className="placeholder">{q}</div>
              <div className="followupsRow">
                <button
                  className="chip"
                  disabled={!!props.disabled}
                  onClick={() => props.onPrefill?.('我补充：')}
                  title="点击后在输入框中填写你的回答（不会自动发送）"
                >
                  填入回答（不自动发送）
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
