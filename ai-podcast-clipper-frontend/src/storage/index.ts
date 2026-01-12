import { OssStorage } from "./oss";
import type { StorageProvider } from "./provider";

let storage: StorageProvider | null = null;

export function getStorage(): StorageProvider {
  storage ??= new OssStorage();
  return storage;
}
