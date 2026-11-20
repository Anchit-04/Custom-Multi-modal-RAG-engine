// File System Access API pieces that TypeScript's DOM library doesn't ship yet.
// Chrome and Edge support them; other browsers fall back to <input type="file">.

interface DirectoryPickerOptions {
  id?: string;
  mode?: 'read' | 'readwrite';
}

interface Window {
  showDirectoryPicker?(options?: DirectoryPickerOptions): Promise<FileSystemDirectoryHandle>;
}
