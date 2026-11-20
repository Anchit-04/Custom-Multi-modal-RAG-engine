/**
 * Extracts a clip's audio as mono samples at the rate speech models expect.
 *
 * Runs on the main thread: AudioContext isn't available inside workers. An
 * OfflineAudioContext created at 16 kHz resamples while it decodes, so the
 * browser does the resampling for us.
 *
 * Limitation: the whole file is read into memory first. Fine for clips up to
 * a few hundred MB; streaming decode (WebCodecs) comes with large libraries in M6.
 */

const MAX_BYTES = 1.5 * 1024 ** 3;

export async function decodeAudio(file: File, sampleRate: number): Promise<Float32Array> {
  if (file.size > MAX_BYTES) {
    throw new Error(`${file.name} is over 1.5 GB; large files need streaming decode (planned for M6)`);
  }

  const bytes = await file.arrayBuffer();
  const context = new OfflineAudioContext(1, 1, sampleRate);
  let audio: AudioBuffer;
  try {
    audio = await context.decodeAudioData(bytes);
  } catch {
    throw new Error(`No audio track found in ${file.name}, or its audio codec isn't supported`);
  }
  return toMono(audio);
}

function toMono(audio: AudioBuffer): Float32Array {
  if (audio.numberOfChannels === 1) return audio.getChannelData(0).slice();

  const mono = new Float32Array(audio.length);
  for (let ch = 0; ch < audio.numberOfChannels; ch++) {
    const data = audio.getChannelData(ch);
    for (let i = 0; i < data.length; i++) mono[i]! += data[i]! / audio.numberOfChannels;
  }
  return mono;
}
