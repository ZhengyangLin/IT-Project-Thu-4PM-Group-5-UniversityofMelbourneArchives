<template>
  <div class="compare-root">
    <div class="compare-col left-col">
      <div class="col-label">Annotation diagram</div>
      <div class="scroll-wrap canvas-wrap" ref="containerRef">
        <canvas ref="fabricCanvasRef"></canvas>
        <div v-if="isLoading" class="canvas-loading">
          <div class="loading-text">Please wait while the picture loads...</div>
        </div>
        <div v-if="loadError && !isLoading" class="canvas-error">
          <el-icon color="red"><Warning /></el-icon>
          <span>Picture Load Failed</span>
          <el-button size="small" @click="loadImage" style="margin-left:8px">Retry</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, onUnmounted } from "vue";
import { Warning } from "@element-plus/icons-vue";
import { Canvas, FabricImage } from "fabric";

const props = defineProps({
  urlImage: String,
  thumbUrlImage: { type: String, default: "" },
});

const fabricCanvasRef = ref(null);
const containerRef = ref(null);
let canvas: Canvas | null = null;
let bgImage: FabricImage | null = null;
let originScale = 1;
let resizeObserver: ResizeObserver | null = null;
let resizeTimer: number | null = null;
let loadingLock = false;
const isLoading = ref(false);
const loadError = ref(false);

const getDrawUrl = () => props.thumbUrlImage || props.urlImage;

const debounce = (fn: Function, delay = 120) => {
  return (...args: any[]) => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => fn(...args), delay);
  };
};

const onContainerResize = debounce(async () => {
  if (!canvas || !bgImage || !containerRef.value) return;
  const cw = containerRef.value.clientWidth - 20;
  const ch = containerRef.value.clientHeight - 20;
  if (cw < 10 || ch < 10) return;
  canvas.setDimensions({ width: cw, height: ch });
  const originW = bgImage.width;
  const originH = bgImage.height;
  originScale = Math.min(cw / originW, ch / originH, 1);
  const dispW = originW * originScale;
  const dispH = originH * originScale;
  bgImage.set({
    scaleX: originScale,
    scaleY: originScale,
    left: (cw - dispW) / 2,
    top: (ch - dispH) / 2,
  });
  canvas.renderAll();
});

const loadImage = async () => {
  const url = getDrawUrl();
  if (!url || loadingLock) return;
  isLoading.value = true;
  loadError.value = false;
  loadingLock = true;
  try {
    if (!canvas) {
      await initCanvas();
      if (!canvas) throw new Error("Canvas initialization failed");
    }
    canvas.clear();
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Request Exception status:${res.status}`);
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const img = await FabricImage.fromURL(blobUrl);
    URL.revokeObjectURL(blobUrl);
    bgImage = img;

    const originW = img.width;
    const originH = img.height;
    const containerDom = containerRef.value;
    const cw = containerDom?.clientWidth - 20 ?? 700;
    const ch = containerDom?.clientHeight - 20 ?? 500;
    originScale = Math.min(cw / originW, ch / originH, 1);
    const dispW = originW * originScale;
    const dispH = originH * originScale;

    canvas.setDimensions({ width: cw, height: ch });
    img.set({
      selectable: false,
      evented: false,
      originX: "left",
      originY: "top",
      scaleX: originScale,
      scaleY: originScale,
      left: (cw - dispW) / 2,
      top: (ch - dispH) / 2,
    });
    canvas.add(img);
    canvas.renderAll();
  } catch (err) {
    console.error(err);
    loadError.value = true;
  } finally {
    isLoading.value = false;
    loadingLock = false;
  }
};


const bindCanvasEvent = () => {
  let isLeftDrag = false;
  canvas!.on("mouse:wheel", (opt) => {
    const delta = opt.e.deltaY;
    let zoom = canvas!.getZoom();
    zoom *= 0.999 ** delta;
    zoom = Math.max(0.2, Math.min(5, zoom));
    canvas!.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
    opt.e.preventDefault();
    opt.e.stopPropagation();
  });

  canvas!.on("mouse:down", (opt) => {
    const evt = opt.e;
    if (evt.button === 1) {
      (canvas as any).isDragging = true;
      (canvas as any).lastPosX = evt.clientX;
      (canvas as any).lastPosY = evt.clientY;
    }
    if (evt.button === 0 && !opt.target) {
      isLeftDrag = true;
      (canvas as any).lastPosX = evt.clientX;
      (canvas as any).lastPosY = evt.clientY;
    }
  });

  canvas!.on("mouse:move", (opt) => {
    const evt = opt.e;
    if ((canvas as any).isDragging) {
      const dx = evt.clientX - (canvas as any).lastPosX;
      const dy = evt.clientY - (canvas as any).lastPosY;
      canvas!.relativePan({ x: dx, y: dy });
      (canvas as any).lastPosX = evt.clientX;
      (canvas as any).lastPosY = evt.clientY;
    }
    if (isLeftDrag) {
      const dx = evt.clientX - (canvas as any).lastPosX;
      const dy = evt.clientY - (canvas as any).lastPosY;
      canvas!.relativePan({ x: dx, y: dy });
      (canvas as any).lastPosX = evt.clientX;
      (canvas as any).lastPosY = evt.clientY;
    }
  });

  canvas!.on("mouse:up", () => {
    (canvas as any).isDragging = false;
    isLeftDrag = false;
  });
};

const initCanvas = async () => {
  await nextTick();
  if (!fabricCanvasRef.value || !containerRef.value) await nextTick();
  if (!fabricCanvasRef.value || !containerRef.value) return;
  const cw = containerRef.value.clientWidth - 20 || 700;
  const ch = containerRef.value.clientHeight - 20 || 500;
  canvas = new Canvas(fabricCanvasRef.value, {
    width: cw,
    height: ch,
    selection: false,
    preserveObjectStacking: true,
  });
  bindCanvasEvent();
  resizeObserver = new ResizeObserver(onContainerResize);
  resizeObserver.observe(containerRef.value);
};

watch(
  () => [props.urlImage, props.thumbUrlImage],
  () => loadImage(),
  { deep: true }
);

defineExpose({ triggerDraw: loadImage });

onUnmounted(() => {
  if (resizeTimer) clearTimeout(resizeTimer);
  if (resizeObserver) resizeObserver.disconnect();
  canvas?.dispose();
});
</script>

<style scoped lang="less">
.compare-root {
  display: grid;
  grid-template-columns: 3fr;
  gap: 12px;
  width: 100%;
  align-items: stretch;
}
.compare-col {
  display: flex;
  flex-direction: column;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  min-height: 420px;
}
.left-col {
  overflow: hidden;
}
.col-label {
  padding: 8px 12px;
  background: #f5f7fa;
  font-weight: 500;
  border-bottom: 1px solid #e4e7ed;
}
.scroll-wrap {
  flex: 1;
  overflow: auto;
  max-height: calc(60vh - 40px);
  padding: 10px;
}
.canvas-wrap {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  position: relative;
  width: 100%;
  max-height: calc(60vh - 40px);
}
.canvas-wrap canvas {
  flex: none;
}
.canvas-loading {
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.75);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
}
.loading-text {
  margin-top: 10px;
  color: #444;
}
.canvas-error {
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
}
</style>
