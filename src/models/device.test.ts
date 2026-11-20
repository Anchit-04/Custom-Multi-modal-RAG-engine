import { describe, expect, it } from 'vitest';
import { isProfileName, pickProfile, type DeviceCaps } from './device';

const GB = 1024 ** 3;
const base: DeviceCaps = { webgpu: true, shaderF16: true, cores: 8, crossOriginIsolated: true };

describe('pickProfile', () => {
  it('uses lite without WebGPU', () => {
    expect(pickProfile({ ...base, webgpu: false, shaderF16: false })).toBe('lite');
  });

  it('uses lite without fp16 shaders (e.g. NVIDIA Maxwell)', () => {
    expect(pickProfile({ ...base, shaderF16: false, gpuVendor: 'nvidia' })).toBe('lite');
  });

  it('uses lite on older Intel integrated GPUs (this dev laptop: Intel UHD, gen-9)', () => {
    expect(
      pickProfile({ ...base, gpuVendor: 'intel', gpuArchitecture: 'gen-9', deviceMemoryGB: 8, maxBufferSize: 2 * GB }),
    ).toBe('lite');
  });

  it('uses lite with very little RAM', () => {
    expect(pickProfile({ ...base, gpuVendor: 'nvidia', deviceMemoryGB: 2 })).toBe('lite');
  });

  it('uses standard on newer Intel GPUs', () => {
    expect(pickProfile({ ...base, gpuVendor: 'intel', gpuArchitecture: 'xe-lpg', deviceMemoryGB: 8 })).toBe('standard');
  });

  it('uses max on a strong discrete GPU', () => {
    expect(
      pickProfile({ ...base, gpuVendor: 'nvidia', gpuArchitecture: 'ada', deviceMemoryGB: 8, maxBufferSize: 4 * GB }),
    ).toBe('max');
  });

  it('uses standard on a strong vendor with small buffers', () => {
    expect(pickProfile({ ...base, gpuVendor: 'amd', deviceMemoryGB: 8, maxBufferSize: 2 * GB })).toBe('standard');
  });
});

describe('isProfileName', () => {
  it('accepts known profiles only', () => {
    expect(isProfileName('max')).toBe(true);
    expect(isProfileName('ultra')).toBe(false);
    expect(isProfileName(null)).toBe(false);
  });
});
