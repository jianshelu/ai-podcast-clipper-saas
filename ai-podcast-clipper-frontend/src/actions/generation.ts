"use server";

import { revalidatePath } from "next/cache";
import { inngest } from "~/inngest/client";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { getStorage } from "~/storage";

export async function processVideo(uploadedFileId: string) {
  const uploadedVideo = await db.uploadedFile.findUniqueOrThrow({
    where: {
      id: uploadedFileId,
    },
    select: {
      uploaded: true,
      id: true,
      userId: true,
    },
  });

  if (uploadedVideo.uploaded) return;

  await inngest.send({
    name: "process-video-events",
    data: { uploadedFileId: uploadedVideo.id, userId: uploadedVideo.userId },
  });

  await db.uploadedFile.update({
    where: {
      id: uploadedFileId,
    },
    data: {
      uploaded: true,
    },
  });

  revalidatePath("/dashboard");
}

export async function getClipPlayUrl(
  clipId: string,
): Promise<{ succes: boolean; url?: string; error?: string }> {
  const session = await auth();
  if (!session?.user?.id) {
    return { succes: false, error: "Unauthorized" };
  }

  try {
    const clip = await db.clip.findUniqueOrThrow({
      where: {
        id: clipId,
        userId: session.user.id,
      },
    });
    const storage = getStorage();
    const signedUrl = await storage.getPresignedGetUrl({
      key: clip.s3Key,
    });

    return { succes: true, url: signedUrl };
  } catch {
    return { succes: false, error: "Failed to generate play URL." };
  }
}

export async function getClipDownloadUrl(
  clipId: string,
): Promise<{ succes: boolean; url?: string; error?: string }> {
  const session = await auth();
  if (!session?.user?.id) {
    return { succes: false, error: "Unauthorized" };
  }

  try {
    const clip = await db.clip.findUniqueOrThrow({
      where: {
        id: clipId,
        userId: session.user.id,
      },
    });

    const filename = clip.s3Key.split("/").pop() ?? `clip-${clip.id}.mp4`;
    const storage = getStorage();
    const signedUrl = await storage.getPresignedGetUrl({
      key: clip.s3Key,
      filename,
    });

    return { succes: true, url: signedUrl };
  } catch {
    return { succes: false, error: "Failed to generate download URL." };
  }
}
