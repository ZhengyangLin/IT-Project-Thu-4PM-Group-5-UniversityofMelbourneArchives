import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import axios from "axios";
import { ElMessage, ElMessageBox } from "element-plus";

const canvases = vi.hoisted(() => [] as Array<{
  getObjects: (type?: string) => any[];
  handlers: Record<string, (event: any) => void>;
  zoomToPoint: ReturnType<typeof vi.fn>;
  relativePan: ReturnType<typeof vi.fn>;
  setDimensions: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  remove: ReturnType<typeof vi.fn>;
}>);

vi.mock("fabric", () => {
  class FakeCanvas {
    objects: unknown[] = [];
    handlers: Record<string, (event: any) => void> = {};
    constructor() { canvases.push(this); }
    add(object: unknown) { this.objects.push(object); }
    clear() { this.objects = []; }
    getObjects(type?: string) {
      if (type !== "rect") return this.objects;
      return this.objects.filter((object) => Boolean((object as { boxData?: unknown }).boxData));
    }
    on(name: string, callback: (event: any) => void) { this.handlers[name] = callback; }
    setDimensions = vi.fn();
    renderAll() {}
    dispose = vi.fn();
    getZoom() { return 1; }
    zoomToPoint = vi.fn();
    relativePan = vi.fn();
    remove = vi.fn();
  }

  class FakeImage {
    width = 100;
    height = 100;
    left = 0;
    top = 0;
    set(values: Record<string, unknown>) { Object.assign(this, values); }
  }

  class FakeRect {
    boxData?: unknown;
    stroke = "#f53f3f";
    strokeWidth = 0.5;
    constructor(values: Record<string, unknown>) { Object.assign(this, values); }
    set(values: Record<string, unknown>) { Object.assign(this, values); }
    setCoords() {}
  }

  return {
    Canvas: FakeCanvas,
    FabricImage: { fromURL: vi.fn(async () => new FakeImage()) },
    Rect: FakeRect,
    FabricText: class {},
    Point: class { constructor(public x: number, public y: number) {} },
  };
});

vi.mock("element-plus", () => ({
  ElMessage: vi.fn(),
  ElMessageBox: { confirm: vi.fn(async () => undefined) },
}));

vi.mock("axios", () => ({ default: { post: vi.fn() } }));

import FabricImg from "@/components/FabricImg.vue";

const fields = {
  architect: {
    value: "Jane Smith",
    confidence: 0.91,
    tokens_bbox: { x0: 0, y0: 0, x1: 20, y1: 20 },
    review_status: "auto_accept",
  },
  date: {
    value: null,
    confidence: 0.2,
    tokens_bbox: { x0: 20, y0: 0, x1: 40, y1: 20 },
    review_status: "needs_review",
  },
  drawing_type: {
    value: "Plan",
    confidence: 0.95,
    tokens_bbox: { x0: 40, y0: 0, x1: 60, y1: 20 },
    review_status: "auto_accept",
  },
};

const wrappers: ReturnType<typeof mount>[] = [];
const resizeCallbacks: ResizeObserverCallback[] = [];
const disconnect = vi.fn();

function mountImage(edit = false, withImage = false, detecBoxs = fields) {
  const wrapper = mount(FabricImg, {
    props: {
      detecBoxs,
      isEdit: edit,
      image_key: "460645/1264425_1984-0047-02865-00001.jpg",
      urlImage: withImage ? "https://example.test/drawing.jpg" : undefined,
    },
    global: {
      stubs: {
        "el-icon": true,
        "el-button": { template: "<button><slot /></button>" },
        "el-input": {
          props: ["modelValue", "size"],
          emits: ["update:modelValue"],
          template: "<input :value='modelValue' @input='$emit(\"update:modelValue\", $event.target.value)' />",
        },
      },
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

describe("FabricImg.vue", () => {
  beforeEach(() => {
    canvases.length = 0;
    resizeCallbacks.length = 0;
    vi.useFakeTimers();
    vi.stubGlobal("ResizeObserver", class {
      constructor(callback: ResizeObserverCallback) { resizeCallbacks.push(callback); }
      observe() {}
      disconnect = disconnect;
    });
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(800);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(600);
  });
  afterEach(() => {
    wrappers.splice(0).forEach((wrapper) => wrapper.unmount());
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("covers OCR display, drawing boxes, editing, review save, and image errors", async () => {
    expect(mountImage().text()).toContain("Architect");
    expect(mountImage().text()).toContain("Jane Smith");
    expect(mountImage().text()).toContain("Confidence:91.0%");
    expect(mountImage().text()).toContain("Final value:-");

    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      blob: async () => new Blob(["image"]),
    })));
    const imageWrapper = mountImage(false, true);
    await flushPromises();
    expect(canvases.at(-1)?.getObjects("rect")).toHaveLength(1);
    expect(imageWrapper.text()).toContain("Drawing type");
    await imageWrapper.findAll(".item")[0].trigger("click");
    expect(canvases.at(-1)?.getObjects("rect")[0].stroke).toBe("#ffc107");
    await imageWrapper.findAll(".item")[0].trigger("click");
    expect(canvases.at(-1)?.getObjects("rect")[0].stroke).toBe("#f53f3f");
    await imageWrapper.findAll(".item")[2].trigger("click");
    expect(imageWrapper.findAll(".item")[2].classes()).not.toContain("active");

    // Real component event handlers must zoom, pan, stop dragging, and resize.
    const canvas = canvases.at(-1)!;
    for (const deltaY of [-10_000, 10_000]) {
      const event = { deltaY, offsetX: 20, offsetY: 30, preventDefault: vi.fn(), stopPropagation: vi.fn() };
      canvas.handlers["mouse:wheel"]({ e: event });
      expect(canvas.zoomToPoint).toHaveBeenLastCalledWith(
        expect.objectContaining({ x: 20, y: 30 }), deltaY < 0 ? 5 : 0.2,
      );
      expect(event.preventDefault).toHaveBeenCalled();
      expect(event.stopPropagation).toHaveBeenCalled();
    }
    for (const button of [0, 1]) {
      canvas.handlers["mouse:down"]({ e: { button, clientX: 10, clientY: 20 } });
      canvas.handlers["mouse:move"]({ e: { clientX: 30, clientY: 50 } });
      expect(canvas.relativePan).toHaveBeenLastCalledWith(expect.objectContaining({ x: 20, y: 30 }));
      canvas.handlers["mouse:up"]({});
      const panCalls = canvas.relativePan.mock.calls.length;
      canvas.handlers["mouse:move"]({ e: { clientX: 40, clientY: 60 } });
      expect(canvas.relativePan).toHaveBeenCalledTimes(panCalls);
    }
    resizeCallbacks[0]([], {} as ResizeObserver);
    resizeCallbacks[0]([], {} as ResizeObserver);
    await vi.advanceTimersByTimeAsync(120);
    expect(canvas.setDimensions).toHaveBeenLastCalledWith({ width: 780, height: 580 });
    const rectangle = canvas.getObjects("rect")[0];
    expect(rectangle.width).toBe(20);
    expect(rectangle.left).toBe(340);
    canvas.handlers["object:removed"]({});
    const label = {};
    const labelBackground = {};
    canvas.handlers["object:removed"]({ target: { labelObj: label, labelBg: labelBackground } });
    expect(canvas.remove).toHaveBeenCalledWith(label);
    expect(canvas.remove).toHaveBeenCalledWith(labelBackground);

    // A changed drawing URL and metadata reload the image and drop stale boxes.
    await imageWrapper.setProps({ thumbUrlImage: "https://example.test/thumbnail.jpg" });
    await flushPromises();
    expect(fetch).toHaveBeenLastCalledWith("https://example.test/thumbnail.jpg");
    await imageWrapper.setProps({ detecBoxs: {
      architect: { value: "New architect", confidence: 1 },
      date: { value: null, tokens_bbox: { x0: 0, y0: 0, x1: 0, y1: 20 } },
    } });
    await flushPromises();
    expect(canvas.getObjects("rect")).toHaveLength(0);
    await imageWrapper.findAll(".item")[0].trigger("click");
    expect(imageWrapper.text()).toContain("New architect");

    const editWrapper = mountImage(true);
    const input = editWrapper.find("input");
    await input.setValue("Updated Architect");
    expect((input.element as HTMLInputElement).value).toBe("Updated Architect");

    vi.mocked(axios.post).mockResolvedValueOnce({
      data: { message: "Saved", result: { image_key: "460645/1264425_1984-0047-02865-00001.jpg" } },
    });
    await editWrapper.findAll("button")[0].trigger("click");
    await flushPromises();
    expect(axios.post).toHaveBeenCalledWith(
      "/Unimelb/ocr-results/manual-review",
      expect.objectContaining({ image_key: "460645/1264425_1984-0047-02865-00001.jpg" }),
    );
    expect(editWrapper.emitted("saved")).toBeTruthy();
    expect(vi.mocked(axios.post).mock.calls.at(-1)?.[1]).toEqual({
      image_key: "460645/1264425_1984-0047-02865-00001.jpg",
      result: { architect: "Updated Architect", date: null, drawing_type: "Plan" },
    });

    for (const message of ["Network Error", "Review is locked"]) {
      vi.mocked(axios.post).mockRejectedValueOnce(new Error(message));
      await editWrapper.find("button").trigger("click");
      await flushPromises();
      expect(ElMessage).toHaveBeenLastCalledWith(expect.objectContaining({ type: "error" }));
      expect(editWrapper.emitted("saved")).toHaveLength(1);
    }
    vi.mocked(ElMessageBox.confirm).mockRejectedValueOnce("cancel");
    const callsBeforeCancel = vi.mocked(axios.post).mock.calls.length;
    await editWrapper.find("button").trigger("click");
    await flushPromises();
    expect(axios.post).toHaveBeenCalledTimes(callsBeforeCancel);
    await editWrapper.setProps({ imageManualReview: 1 });
    expect(editWrapper.text()).toContain("Reviewed");
    expect(editWrapper.text()).not.toContain("Confidence:");

    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, blob: async () => new Blob() })));
    const failedWrapper = mountImage(false, true);
    await flushPromises();
    expect(failedWrapper.text()).toContain("Picture Load Failed");
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, blob: async () => new Blob(["retry"]) })));
    await failedWrapper.find("button").trigger("click");
    await flushPromises();
    expect(failedWrapper.text()).not.toContain("Picture Load Failed");
    expect(canvases.at(-1)?.getObjects("rect")).toHaveLength(1);

    expect(mountImage(false, false, {}).text()).toContain("No labeled data available.");
    const loadedCanvases = [...canvases];
    wrappers.splice(0).forEach((wrapper) => wrapper.unmount());
    loadedCanvases.forEach((item) => expect(item.dispose).toHaveBeenCalledOnce());
    expect(disconnect).toHaveBeenCalledTimes(loadedCanvases.length);
  });
});
