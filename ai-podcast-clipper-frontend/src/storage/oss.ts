import OSS from "ali-oss";
import { env } from "~/env";
import type { PresignGetOptions, PresignPutOptions, StorageProvider } from "./provider";

type OssSignatureOptions = {
  method?: "PUT" | "GET";
  expires?: number;
  response?: Record<string, string>;
};

type OssListResult = {
  objects?: { name: string }[];
  isTruncated?: boolean;
  nextMarker?: string;
};

type OssClient = {
  signatureUrl: (key: string, options: OssSignatureOptions) => string;
  list: (options: {
    prefix: string;
    marker?: string;
    "max-keys"?: number;
  }) => Promise<OssListResult>;
};

type OssClientConstructor = new (options: {
  accessKeyId: string;
  accessKeySecret: string;
  region: string;
  endpoint: string;
  bucket: string;
  secure?: boolean;
}) => OssClient;

let cachedClient: OssClient | null = null;

function getClient(): OssClient {
  if (cachedClient) return cachedClient;
  const OssCtor = OSS as unknown as OssClientConstructor;
  cachedClient = new OssCtor({
    accessKeyId: env.OSS_ACCESS_KEY_ID,
    accessKeySecret: env.OSS_ACCESS_KEY_SECRET,
    region: env.OSS_REGION,
    endpoint: env.OSS_ENDPOINT,
    bucket: env.OSS_BUCKET,
    secure: true,
  });
  return cachedClient;
}

export class OssStorage implements StorageProvider {
  async getPresignedPutUrl({ key, expiresInSeconds = 600 }: PresignPutOptions) {
    const client = getClient();
    return client.signatureUrl(key, {
      method: "PUT",
      expires: expiresInSeconds,
    });
  }

  async getPresignedGetUrl({
    key,
    expiresInSeconds = 3600,
    filename,
  }: PresignGetOptions) {
    const client = getClient();
    return client.signatureUrl(key, {
      expires: expiresInSeconds,
      response: filename
        ? {
            "content-disposition": `attachment; filename=\"${filename}\"`,
          }
        : undefined,
    });
  }

  async listKeysByPrefix(prefix: string) {
    const client = getClient();
    const keys: string[] = [];
    let nextMarker: string | undefined;
    let truncated = true;

    while (truncated) {
      const result = await client.list({
        prefix,
        marker: nextMarker,
        "max-keys": 1000,
      });
      keys.push(...(result.objects?.map((item) => item.name) ?? []));
      truncated = Boolean(result.isTruncated);
      nextMarker = result.nextMarker;
    }

    return keys;
  }
}
