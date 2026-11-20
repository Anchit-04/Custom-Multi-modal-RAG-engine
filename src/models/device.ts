/**
 * Hardware check and profile choice.
 *
 * The same app runs everywhere; only model sizes change. probeDevice() reads
 * what the browser exposes (it works on the main thread and in workers), and
 * pickProfile() is a pure function so the policy is unit-tested.
 */

export type ProfileName = 'lite' | 'standard' | 'max';
export const PROFILE_NAMES: readonly ProfileName[] = ['lite', 'standard', 'max'];

export interface DeviceCaps {
  webgpu: boolean;
  /** Half-precision shaders: halves model memory and speeds up inference. */
  shaderF16: boolean;
  gpuVendor?: string;
  gpuArchitecture?: string;
  /** Largest single GPU buffer, in bytes. */
  maxBufferSize?: number;
  /** navigator.deviceMemory: approximate RAM in GB, capped at 8 by Chrome. */
  deviceMemoryGB?: number;
  cores: number;
  /** Needed for multithreaded WASM, the CPU fallback. */
  crossOriginIsolated: boolean;
}

export async function probeDevice(): Promise<DeviceCaps> {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const caps: DeviceCaps = {
    webgpu: false,
    shaderF16: false,
    deviceMemoryGB: nav.deviceMemory,
    cores: nav.hardwareConcurrency ?? 1,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
  };

  if (!nav.gpu) return caps;
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return caps;
    caps.webgpu = true;
    caps.shaderF16 = adapter.features.has('shader-f16');
    caps.gpuVendor = adapter.info?.vendor || undefined;
    caps.gpuArchitecture = adapter.info?.architecture || undefined;
    caps.maxBufferSize = adapter.limits.maxBufferSize;
  } catch {
    // WebGPU present but unusable (blocklisted driver, etc.): fall back to CPU.
  }
  return caps;
}

const GB = 1024 ** 3;

/**
 * Policy:
 *   lite     no WebGPU, no fp16, an older Intel integrated GPU (gen-*), or little RAM
 *   max      fp16 GPU from NVIDIA, AMD or Apple, plus at least 8 GB RAM and 4 GB buffers
 *   standard everything in between (newer Intel Xe/Arc, mid-range GPUs)
 */
export function pickProfile(caps: DeviceCaps): ProfileName {
  if (!caps.webgpu || !caps.shaderF16) return 'lite';
  if (caps.deviceMemoryGB !== undefined && caps.deviceMemoryGB < 4) return 'lite';

  const vendor = caps.gpuVendor?.toLowerCase() ?? '';
  const arch = caps.gpuArchitecture?.toLowerCase() ?? '';
  if (vendor === 'intel' && arch.startsWith('gen')) return 'lite';

  const strongVendor = vendor === 'nvidia' || vendor === 'amd' || vendor === 'apple';
  const enoughRam = (caps.deviceMemoryGB ?? 0) >= 8;
  const bigBuffers = (caps.maxBufferSize ?? 0) >= 4 * GB;
  if (strongVendor && enoughRam && bigBuffers) return 'max';

  return 'standard';
}

export function isProfileName(value: unknown): value is ProfileName {
  return typeof value === 'string' && (PROFILE_NAMES as readonly string[]).includes(value);
}
