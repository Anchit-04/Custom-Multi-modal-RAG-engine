/**
 * Getting footage into the app: a whole folder (Chrome/Edge) or picked files.
 * Files are read in place; nothing is copied or uploaded.
 */
import type { ClipId } from '../core/types';

const VIDEO_EXTENSIONS = /\.(mp4|m4v|mov|webm|mkv)$/i;

export interface PickedFile {
  file: File;
  /** Path relative to the picked folder, or just the file name. */
  path: string;
}

export function canPickFolder(): boolean {
  return typeof window.showDirectoryPicker === 'function';
}

/** Opens the folder picker and returns every video inside it, recursively. Null if the user cancels. */
export async function pickFolder(): Promise<PickedFile[] | null> {
  if (!window.showDirectoryPicker) throw new Error('This browser cannot open folders; use "Add files" instead.');
  try {
    const dir = await window.showDirectoryPicker({ id: 'framegrep-footage', mode: 'read' });
    const files = await collectVideos(dir, '');
    return files.sort((a, b) => a.path.localeCompare(b.path));
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return null;
    throw err;
  }
}

async function collectVideos(dir: FileSystemDirectoryHandle, prefix: string): Promise<PickedFile[]> {
  const found: PickedFile[] = [];
  for await (const handle of dir.values()) {
    const path = prefix ? `${prefix}/${handle.name}` : handle.name;
    if (handle.kind === 'directory') {
      found.push(...(await collectVideos(handle as FileSystemDirectoryHandle, path)));
    } else if (VIDEO_EXTENSIONS.test(handle.name)) {
      found.push({ file: await (handle as FileSystemFileHandle).getFile(), path });
    }
  }
  return found;
}

/** Files chosen through <input type="file">. */
export function fromFileList(list: FileList | null): PickedFile[] {
  if (!list) return [];
  return Array.from(list)
    .filter((file) => VIDEO_EXTENSIONS.test(file.name))
    .map((file) => ({ file, path: file.webkitRelativePath || file.name }));
}

/** Stable id: the same file picked twice maps to the same clip. */
export function clipIdFor(picked: PickedFile): ClipId {
  return `${picked.path}|${picked.file.size}|${picked.file.lastModified}`;
}
