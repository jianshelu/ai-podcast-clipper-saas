"use server";

import { getStorage } from "~/storage";
import { auth } from "~/server/auth";
import { v4 as uuidv4 } from "uuid";
import { db } from "~/server/db";

export async function generateUploadUrl(fileInfo: {
  filename: string;
  contentType: string;
}): Promise<{
  success: boolean;
  signedUrl: string;
  key: string;
  uploadedFileId: string;
}> {
  const session = await auth();
  if (!session) throw new Error("Unauthorized");

  const storage = getStorage();

  const fileExtension = fileInfo.filename.split(".").pop() ?? "";

  const jobId = uuidv4();
  const key = `jobs/${jobId}/inputs/original.${fileExtension}`;
  const signedUrl = await storage.getPresignedPutUrl({ key });

  const uploadedFileDbRecord = await db.uploadedFile.create({
    data: {
      userId: session.user.id,
      s3Key: key,
      displayName: fileInfo.filename,
      uploaded: false,
    },
    select: {
      id: true,
    },
  });

  return {
    success: true,
    signedUrl,
    key,
    uploadedFileId: uploadedFileDbRecord.id,
  };
}
