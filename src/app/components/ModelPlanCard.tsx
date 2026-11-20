import type { ProfileName } from '../../models/device';
import type { ModelChoice, ModelPlan } from '../../models/profiles';

interface Props {
  profile: ProfileName;
  plan: ModelPlan;
}

const ROWS: Array<[keyof ModelPlan, string]> = [
  ['speech', 'Speech'],
  ['visual', 'Visual vectors'],
  ['ocr', 'Screen text'],
  ['rerank', 'Re-ranking'],
  ['captions', 'Captions'],
  ['textEmbed', 'Text meaning'],
];

function Choice({ choice }: { choice: ModelChoice | null }) {
  if (!choice) return <span className="muted">off on this profile</span>;
  return (
    <>
      {choice.label} <span className="muted">· {choice.milestone}</span>
    </>
  );
}

export function ModelPlanCard({ profile, plan }: Props) {
  return (
    <section className="card">
      <h2>Models for “{profile}”</h2>
      <dl className="facts">
        {ROWS.map(([key, label]) => (
          <div key={key} className="fact-row">
            <dt>{label}</dt>
            <dd>
              <Choice choice={plan[key]} />
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
