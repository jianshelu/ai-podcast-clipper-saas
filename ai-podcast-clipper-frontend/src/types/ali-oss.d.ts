declare module "ali-oss" {
  export type SignatureOptions = {
    method?: "PUT" | "GET";
    expires?: number;
    response?: Record<string, string>;
  };

  export type ListResult = {
    objects?: { name: string }[];
    isTruncated?: boolean;
    nextMarker?: string;
  };

  export type ClientOptions = {
    accessKeyId: string;
    accessKeySecret: string;
    region: string;
    endpoint: string;
    bucket: string;
    secure?: boolean;
  };

  export default class OSS {
    constructor(options: ClientOptions);
    signatureUrl(key: string, options: SignatureOptions): string;
    list(options: {
      prefix: string;
      marker?: string;
      "max-keys"?: number;
    }): Promise<ListResult>;
  }
}
