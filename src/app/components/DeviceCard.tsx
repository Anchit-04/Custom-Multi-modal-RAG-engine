import type { DeviceCaps, ProfileName } from '../../models/device';

interface Props {
  caps: DeviceCaps;
  profile: ProfileName;
  overridden: boolean;
}

function formatGB(bytes?: number): string {
  return bytes === undefined ? 'unknown' : `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export function DeviceCard({ caps, profile, overridden }: Props) {
  const gpu = caps.webgpu
    ? [caps.gpuVendor, caps.gpuArchitecture].filter(Boolean).join(' · ') || 'WebGPU (unnamed adapter)'
    : 'No WebGPU: CPU fallback';

  return (
    <section className="card">
      <h2>This device</h2>
      <dl className="facts">
        <dt>Profile</dt>
        <dd>
          <span className={`pill pill-${profile}`}>{profile}</span>
          {overridden && <span className="muted"> (set by ?profile=)</span>}
        </dd>
        <dt>GPU</dt>
        <dd>{gpu}</dd>
        <dt>fp16 shaders</dt>
        <dd>{caps.shaderF16 ? 'yes' : 'no'}</dd>
        <dt>Max GPU buffer</dt>
        <dd>{formatGB(caps.maxBufferSize)}</dd>
        <dt>RAM (reported)</dt>
        <dd>{caps.deviceMemoryGB === undefined ? 'unknown' : `≥ ${caps.deviceMemoryGB} GB`}</dd>
        <dt>CPU threads</dt>
        <dd>{caps.cores}</dd>
        <dt>Multithreaded WASM</dt>
        <dd>{caps.crossOriginIsolated ? 'available' : 'unavailable (page not cross-origin isolated)'}</dd>
      </dl>
    </section>
  );
}
