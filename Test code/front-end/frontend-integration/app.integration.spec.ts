import { expect, test, type APIRequestContext } from "@playwright/test";

const apiBase = "http://127.0.0.1:8000";

type CoverageFile = {
  statementMap: Record<string, { start: { line: number }; end: { line: number } }>;
  s: Record<string, number>;
};

function lineCoverage(file: CoverageFile) {
  const lines = new Map<number, boolean>();
  for (const [statementId, statement] of Object.entries(file.statementMap)) {
    for (let line = statement.start.line; line <= statement.end.line; line += 1) {
      lines.set(line, (lines.get(line) ?? false) || file.s[statementId] > 0);
    }
  }
  const covered = [...lines.values()].filter(Boolean).length;
  return { covered, total: lines.size, percent: (covered / lines.size) * 100 };
}

async function getFirstImage(request: APIRequestContext) {
  const response = await request.get(`${apiBase}/Unimelb/ImageSelect`);
  expect(response.ok()).toBe(true);
  const body = await response.json();
  expect(body.result.total).toBeGreaterThan(0);
  return { body, image: body.result.images[0] };
}

test("covers the complete frontend-backend workflow", async ({ page, request }) => {
  const { body, image } = await getFirstImage(request);
  await page.goto("/");
  await expect(page.getByText("Image task list")).toBeVisible();
  await expect(page.getByText(String(image.node_id))).toBeVisible();
  await expect(page.getByText(image.image_name)).toBeVisible();

  await page.locator(".stat-card-fail").click();
  await expect(page.getByText("Failure").first()).toBeVisible();
  const folder = page.locator(".folder-item").nth(1);
  await folder.click();
  await expect(folder).toHaveClass(/file-on/);
  expect(body.result.folders).toBeTruthy();

  const imageResult = await page.evaluate(async (imageUrl) => {
    const response = await fetch(imageUrl);
    return {
      status: response.status,
      contentType: response.headers.get("content-type") || "",
    };
  }, image.image_url);
  expect(imageResult.status).toBe(200);
  expect(imageResult.contentType).toContain("image/");

  const reviewResult = await page.evaluate(
    async ({ url, imageKey }) => {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_key: imageKey,
          result: {
            drawing_number: "S24",
            project_name: "STATE GOVERNMENT OFFICES",
            drawing_title: "MULLION DETAILS",
            architect: "YUNCKEN ARCHITECTS",
            draughtsperson: null,
            date: "4.11.63",
            scale: "1:4",
            drawing_type: "Detail",
          },
        }),
      });
      return { status: response.status, body: await response.json() };
    },
    {
      url: `${apiBase}/Unimelb/ocr-results/manual-review`,
      imageKey: image.image_key,
    },
  );
  expect([200, 409]).toContain(reviewResult.status);
  expect(reviewResult.body).toBeTruthy();

  await page.addInitScript(() => {
    Object.defineProperty(window, "showSaveFilePicker", {
      value: async () => ({
        createWritable: async () => ({
          write: async () => undefined,
          close: async () => undefined,
        }),
      }),
    });
  });
  await page.reload();
  const firstRow = page.locator(".img-row").nth(1);
  await firstRow.locator(".el-checkbox").click();
  const exportRequest = page.waitForRequest(
    (request) =>
      new URL(request.url()).pathname === "/Unimelb/ocr-results/export" &&
      request.method() === "POST",
  );
  await page.getByRole("button", { name: "Export" }).click();
  await expect(page.getByRole("button", { name: "OK" })).toBeVisible();
  await page.getByRole("button", { name: "OK" }).click();
  const outgoing = await exportRequest;
  expect(outgoing.postDataJSON().image_keys).toHaveLength(1);

  if (process.env.INTEGRATION_COVERAGE === "1") {
    const coverage = await page.evaluate(() =>
      (window as unknown as { __coverage__?: Record<string, CoverageFile> }).__coverage__ ?? {},
    );
    const appEntry = Object.entries(coverage).find(([file]) => file.endsWith("/src/App.vue"));
    expect(appEntry).toBeTruthy();
    const measured = lineCoverage(appEntry![1]);
    console.log(
      `INTEGRATION COVERAGE IT-001 App.vue: ${measured.percent.toFixed(2)}% (${measured.covered}/${measured.total} lines)`,
    );
    expect(measured.percent).toBeGreaterThanOrEqual(60);
  }
});
