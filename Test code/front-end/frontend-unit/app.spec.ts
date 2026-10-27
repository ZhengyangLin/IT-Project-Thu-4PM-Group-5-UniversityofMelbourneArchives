import { afterEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import axios from "axios";
import { ElMessage, ElMessageBox } from "element-plus";
import App from "@/App.vue";

vi.mock("element-plus", () => ({
  ElMessage: Object.assign(vi.fn(), { warning: vi.fn(), error: vi.fn() }),
  ElMessageBox: { confirm: vi.fn(async () => undefined) },
}));

vi.mock("axios", () => {
  const request = vi.fn();
  request.get = vi.fn();
  request.post = vi.fn();
  return { default: request };
});

const taskData = {
  total: 2,
  folders: { YFA_STATE_GOVERNMENT: 2 },
  status_counts: { 0: 1, 1: 0, 2: 1, 3: 0 },
  image_name: "",
  image_key: "",
  images: [
    {
      image_key: "460645/1264425_1984-0047-02865-00001.jpg",
      image_name: "1264425_1984-0047-02865-00001.jpg",
      folder: "YFA_STATE_GOVERNMENT",
      node_id: "460645",
      status: 0,
      manual_reviewed: 0,
    },
    {
      image_key: "455992/1238196_1984-0047-02635-00001.jpg",
      image_name: "1238196_1984-0047-02635-00001.jpg",
      folder: "YFA_STATE_GOVERNMENT",
      node_id: "455992",
      status: 2,
      manual_reviewed: 1,
      result: {},
    },
  ],
};

const DialogStub = {
  props: ["modelValue", "title"],
  template: `<div v-if="modelValue" class="mock-dialog"><h2>{{ title }}</h2><slot /><slot name="footer" /></div>`,
};

const wrappers: ReturnType<typeof mount>[] = [];
const draw = vi.fn();

function mountApp() {
  const wrapper = mount(App, {
    global: {
      stubs: {
        FabricImg: {
          name: "FabricImg",
          props: ["detecBoxs", "image_key", "imageManualReview"],
          emits: ["saved"],
          setup: (_props: unknown, { expose }: { expose: (value: unknown) => void }) => {
            expose({ triggerDraw: draw });
          },
          template: "<div class='mock-fabric-img' />",
        },
        "el-icon": true,
        "el-tooltip": true,
        "el-form": { template: "<div><slot /></div>" },
        "el-pagination": {
          name: "PaginationStub",
          emits: ["size-change", "current-change"],
          template: "<div class='mock-pagination' />",
        },
        "el-dialog": DialogStub,
        "el-button": { template: "<button v-bind='$attrs'><slot /></button>" },
        "el-checkbox": {
          template: "<button class='mock-checkbox' @click='$emit(\"click\")'></button>",
          emits: ["click"],
        },
      },
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

describe("App.vue", () => {
  afterEach(() => {
    wrappers.splice(0).forEach((wrapper) => wrapper.unmount());
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("covers the task-list workflow, actions, empty state, and load errors", async () => {
    vi.useFakeTimers();
    vi.mocked(axios.get).mockResolvedValue({ data: { result: taskData } });
    vi.mocked(axios.post).mockResolvedValue({ data: { message: "Queued" } });

    const wrapper = mountApp();
    await flushPromises();

    expect(wrapper.text()).toContain("1264425_1984-0047-02865-00001.jpg");
    expect(wrapper.text()).toContain("Pending");
    expect(wrapper.text()).toContain("Success");
    expect(wrapper.text()).toContain("YFA_STATE_GOVERNMENT");
    expect(wrapper.text()).toContain("50%");

    await wrapper.find(".stat-card-success").trigger("click");
    expect(wrapper.text()).not.toContain("1264425_1984-0047-02865-00001.jpg");
    expect(wrapper.text()).toContain("1238196_1984-0047-02635-00001.jpg");
    await wrapper.find(".stat-card-total").trigger("click");
    await wrapper.findAll(".folder-item")[1].trigger("click");
    await wrapper.findAll(".folder-item")[1].trigger("click");

    await wrapper.findAll(".mock-checkbox")[1].trigger("click");
    await wrapper.findAll("button").find((button) => button.text() === "Batch recheck")?.trigger("click");
    await flushPromises();
    expect(axios.post).toHaveBeenCalledWith(
      "/Unimelb/ocr-tasks/rerun",
      { image_keys: [taskData.images[0].image_key] },
    );

    const checkButton = wrapper.findAll("button").find((button) => button.text() === "Check");
    await checkButton?.trigger("click");
    expect(wrapper.text()).toContain("Are you sure you want to check this image?");
    await wrapper.findAll("button").find((button) => button.text() === "Cancel")?.trigger("click");

    const completedRow = wrapper.findAll(".img-box-list > .img-row")[1];
    const reviewButton = completedRow!.findAll("button").find((button) => button.text() === "Manual review");
    await reviewButton!.trigger("click");
    expect(wrapper.text()).toContain("Extraction results");

    // A saved review reloads the selected record and refreshes the dialog.
    const review = wrapper.findComponent({ name: "FabricImg" });
    const updatedData = structuredClone(taskData);
    updatedData.images[1].manual_reviewed = 1;
    updatedData.images[1].result = { drawing_number: { value: "603/A2" } };
    vi.mocked(axios.get).mockResolvedValueOnce({ data: { result: updatedData } });
    review.vm.$emit("saved", taskData.images[1].image_key);
    await flushPromises();
    expect(review.props("detecBoxs")).toEqual(updatedData.images[1].result);
    wrapper.findAllComponents(DialogStub)[1].vm.$emit("open");
    await flushPromises();
    expect(draw).toHaveBeenCalled();

    const button = (text: string) => {
      const found = wrapper.findAll("button").find((item) => item.text() === text);
      expect(found, `Button ${text} should exist`).toBeDefined();
      return found!;
    };

    // Selecting every row toggles off again; no selection must not send a rerun.
    await wrapper.findAll(".mock-checkbox")[0].trigger("click");
    await wrapper.findAll(".mock-checkbox")[0].trigger("click");
    await button("Batch recheck").trigger("click");
    expect(ElMessage.warning).toHaveBeenCalled();
    await wrapper.findAll(".mock-checkbox")[1].trigger("click");
    await wrapper.findAll(".mock-checkbox")[1].trigger("click");
    await wrapper.findAll(".mock-checkbox")[1].trigger("click");

    // Exercise both rerun failure responses through the batch action.
    for (const message of ["Network Error", "Recheck unavailable"]) {
      vi.mocked(axios.post).mockRejectedValueOnce(new Error(message));
      await button("Batch recheck").trigger("click");
      await flushPromises();
      expect(ElMessage).toHaveBeenLastCalledWith(expect.objectContaining({ type: "error" }));
    }

    // Pagination changes which rows are visible and resets after a size change.
    const pagination = wrapper.findComponent({ name: "PaginationStub" });
    pagination.vm.$emit("size-change", 1);
    await flushPromises();
    expect(wrapper.findAll(".img-box-list > .img-row")).toHaveLength(1);
    pagination.vm.$emit("current-change", 2);
    await flushPromises();
    expect(wrapper.text()).toContain(taskData.images[1].image_name);
    expect(wrapper.text()).not.toContain(taskData.images[0].image_name);
    pagination.vm.$emit("size-change", 20);
    await wrapper.find(".folder-panel .panel-title").trigger("click");

    // Both task-start buttons pass the correct force flag to the backend.
    for (const [index, force] of [[0, true], [1, false]] as const) {
      await wrapper.findAll(".float-ball-wrap")[index].trigger("click");
      await flushPromises();
      expect(axios.post).toHaveBeenLastCalledWith("/Unimelb/ocr-tasks", { force });
    }
    for (const message of ["Network Error", "Task already exists"]) {
      vi.mocked(axios.post).mockRejectedValueOnce(new Error(message));
      await wrapper.findAll(".float-ball-wrap")[0].trigger("click");
      await flushPromises();
      expect(ElMessage).toHaveBeenLastCalledWith(expect.objectContaining({ type: "error" }));
    }
    const callsBeforeCancel = vi.mocked(axios.post).mock.calls.length;
    vi.mocked(ElMessageBox.confirm).mockRejectedValueOnce("cancel");
    await wrapper.findAll(".float-ball-wrap")[0].trigger("click");
    await flushPromises();
    expect(axios.post).toHaveBeenCalledTimes(callsBeforeCancel);

    // Export creates a download with the server filename and selected key.
    const downloads: Array<{ name: string; href: string }> = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function () {
      downloads.push({ name: this.download, href: this.href });
    });
    vi.mocked(axios).mockResolvedValueOnce({
      status: 200,
      data: new Blob(["workbook"], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }),
      headers: { "content-disposition": 'attachment; filename="results.xlsx"' },
    });
    await button("Export").trigger("click");
    await flushPromises();
    expect(axios).toHaveBeenLastCalledWith(expect.objectContaining({
      url: "/Unimelb/ocr-results/export",
      data: { image_keys: [taskData.images[0].image_key] },
      responseType: "blob",
    }));
    expect(downloads[0]).toEqual({ name: "results.xlsx", href: "blob:uma-test" });

    vi.mocked(axios).mockResolvedValueOnce({ status: 500, statusText: "Export unavailable" });
    await button("Export").trigger("click");
    await flushPromises();
    expect(ElMessage.error).toHaveBeenLastCalledWith("Export unavailable");
    vi.mocked(axios).mockResolvedValueOnce({
      status: 200,
      data: { type: "application/json", text: async () => JSON.stringify({ message: "No results" }) },
    });
    await button("Export").trigger("click");
    await flushPromises();
    expect(ElMessage.error).toHaveBeenLastCalledWith("No results");
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(axios).mockRejectedValueOnce(new Error("Download unavailable"));
    await button("Export").trigger("click");
    await flushPromises();
    expect(ElMessage.error).toHaveBeenLastCalledWith("Export failed. Please try again.");
    vi.mocked(ElMessageBox.confirm).mockRejectedValueOnce("cancel");
    const exportCalls = vi.mocked(axios).mock.calls.length;
    await button("Export").trigger("click");
    await flushPromises();
    expect(axios).toHaveBeenCalledTimes(exportCalls);

    // Polling refreshes data, reports a failure once, and recovers afterwards.
    await vi.advanceTimersByTimeAsync(5000);
    expect(axios.get).toHaveBeenCalledWith("/Unimelb/ImageSelect");
    for (const message of ["Network Error", "Polling failed"]) {
      vi.mocked(axios.get).mockRejectedValueOnce(new Error(message));
      await vi.advanceTimersByTimeAsync(5000);
      expect(ElMessage).toHaveBeenLastCalledWith(expect.objectContaining({ type: "error" }));
      const warnings = vi.mocked(ElMessage).mock.calls.length;
      vi.mocked(axios.get).mockRejectedValueOnce(new Error(message));
      await vi.advanceTimersByTimeAsync(5000);
      expect(ElMessage).toHaveBeenCalledTimes(warnings);
      await vi.advanceTimersByTimeAsync(5000);
    }

    vi.mocked(axios.get).mockResolvedValueOnce({
      data: { result: { ...taskData, total: 0, images: [], folders: {}, status_counts: { 0: 0, 1: 0, 2: 0, 3: 0 } } },
    });
    const emptyWrapper = mountApp();
    await flushPromises();
    expect(emptyWrapper.text()).toContain("No image task data available at present.");
    await emptyWrapper.find(".mock-checkbox").trigger("click");
    await emptyWrapper.findAll("button").find((item) => item.text() === "Export")!.trigger("click");
    expect(ElMessage).toHaveBeenLastCalledWith({ message: "There are no exportable data.", type: "warning" });

    vi.mocked(axios.get).mockRejectedValueOnce(new Error("Network Error"));
    const errorWrapper = mountApp();
    await flushPromises();
    expect(errorWrapper.exists()).toBe(true);
    expect(errorWrapper.text()).toContain("Image task list");
    vi.mocked(axios.get).mockRejectedValueOnce(new Error("Access denied"));
    mountApp();
    await flushPromises();
    expect(ElMessage).toHaveBeenLastCalledWith({ message: "Query of task status failed:Access denied", type: "error" });

    wrappers.forEach((item) => item.unmount());
    wrappers.length = 0;
    const requests = vi.mocked(axios.get).mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(axios.get).toHaveBeenCalledTimes(requests);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:uma-test");
  });
});
