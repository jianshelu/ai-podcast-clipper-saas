export type PresignPutOptions = {
  key: string;
  expiresInSeconds?: number;
};

export type PresignGetOptions = {
  key: string;
  expiresInSeconds?: number;
  filename?: string;
};

export interface StorageProvider {
  getPresignedPutUrl(options: PresignPutOptions): Promise<string>;
  getPresignedGetUrl(options: PresignGetOptions): Promise<string>;
  listKeysByPrefix(prefix: string): Promise<string[]>;
}
