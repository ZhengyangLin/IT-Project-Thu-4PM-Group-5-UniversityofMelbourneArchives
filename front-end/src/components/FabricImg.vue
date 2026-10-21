<template>
  <div style="text-align: right; padding-bottom: 5px" v-if="isEdit">
    <el-button type="success" @click="submit()">Review</el-button>
  </div>
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
          <el-button
            size="small"
            @click="loadImageAndBox"
            style="margin-left: 8px"
            >Retry</el-button
          >
        </div>
      </div>
    </div>
    <div class="compare-col right-col">
      <div class="col-label">Annotation information</div>
      <div class="list-message">
        <div
          class="item"
          v-for="item in detecBoxEntries"
          :key="item.fieldKey"
          :class="{ active: activeFieldKey === item.fieldKey }"
          @click="handleClickItem(item.fieldKey, item)"
        >
          <div class="field-name">
            {{ fieldNameMap[item.fieldKey] || item.fieldKey }}
          </div>
          <div class="field-value" :title="String(item.value ?? '')">
            <!-- && (fieldNameMap[item.fieldKey] || item.fieldKey) === 'Figure number' -->
            <span v-if="isEdit" >
              Final value:
              <el-input
                :model-value="getFieldValue(item.fieldKey)"
                @update:model-value="setFieldValue(item.fieldKey, $event)"
                placeholder=""
                size="small"
              />
            </span>
            <span v-else :style="item.value ? '' : 'color:red'">
              Final value:{{ item.value ?? "-" }}</span
            >
          </div>
          <div v-if="props.imageManualReview == 0" class="field-meta">
            <div class="conf">
              Confidence:{{ formatConfidence(item.confidence) }}
            </div>
            <div class="conf">
              Original text of OCR evidence:{{ item.verbatim }}
            </div>
            <div class="conf">
              The normalized or corrected value of the original text:{{
                item.corrected
              }}
            </div>
            <div class="conf">
              The specific basis for modifying OCR characters:{{
                item.correction_basis
              }}
            </div>
            <div class="conf">Field source: {{ item.source }}</div>
            <div class="conf">
              Structured reason code for field impermanent value:{{
                item.reason
              }}
            </div>
            <div class="conf">Remarks:{{ item.notes }}</div>

            <span
              class="status-tag"
              :class="{
                'tag-accept': item.review_status === 'auto_accept',
                'tag-review': item.review_status === 'needs_review',
                'tag-red': !item.value,
              }"
            >
              {{ item.value ? item.review_status : "No data was found" }}
            </span>
          </div>
          <div v-else-if="props.imageManualReview === 1" class="field-meta">
            <span class="status-tag tag-reviewed">Reviewed</span>
          </div>
        </div>
        <div v-if="detecBoxEntries.length === 0" class="empty-item">
          No labeled data available.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, onUnmounted, computed, unref } from "vue";
import { Warning } from "@element-plus/icons-vue";
import { Canvas, FabricImage, Rect, FabricText, Point } from "fabric";
import type { FabricObject } from "fabric";
import { ElMessageBox, ElMessage } from "element-plus";
import axios from "axios";
interface TokenBBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface DetecBoxItem {
  tokens_bbox?: TokenBBox;
  confidence?: number;
  value?: string | number | null;
  verbatim?: string;
  corrected?: string;
  correction_basis?: string;
  source?: string;
  reason?: string;
  notes?: string;
  review_status?: string;
}
type DetecBoxMap = Record<string, DetecBoxItem>;
type DetecBoxEntry = DetecBoxItem & { fieldKey: string };
interface NormalizedBox extends DetecBoxItem {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label?: string | number | null;
  conf?: number;
  tokens_bbox?: TokenBBox;
}
type BoxRect = Rect & {
  boxData?: NormalizedBox;
  labelObj?: FabricText;
  labelBg?: Rect;
  _originStroke?: string;
  _originStrokeWidth?: number;
};
const props = withDefaults(
  defineProps<{
    urlImage?: string;
    thumbUrlImage?: string;
    detecBoxs?: DetecBoxMap;
    image_key?: string;
    confThreshold?: number;
    isEdit?: boolean;
    imageManualReview?: number;
  }>(),
  {
    urlImage: undefined,
    thumbUrlImage: "",
    detecBoxs: () => ({}),
    image_key: "",
    confThreshold: 0.3,
    isEdit: false,
    imageManualReview: 0,
  }
);
const emit = defineEmits<{
  saved: [imageKey: string];
}>();
enum Api {
  manualReview = "http://127.0.0.1:8000/Unimelb/ocr-results/manual-review",
}
const fabricCanvasRef = ref<HTMLCanvasElement | null>(null);
const containerRef = ref<HTMLDivElement | null>(null);
let canvas: Canvas | null = null;
let bgImage: FabricImage | null = null;
let originScale = 1;
let resizeObserver: ResizeObserver | null = null;
let resizeTimer: ReturnType<typeof setTimeout> | null = null;
let loadingLock = false;
const isLoading = ref<boolean>(false);
const loadError = ref<boolean>(false);

const localDetecBoxs = ref<DetecBoxMap>({});
const getDrawUrl = () => props.thumbUrlImage || props.urlImage;
const getBoxRects = (cv: Canvas): BoxRect[] =>
  (cv.getObjects("rect") as FabricObject[]).filter(
    (o) => (o as BoxRect).boxData
  ) as unknown as BoxRect[];

const debounce = (fn: (...args: any[]) => void, delay = 120) => {
  return (...args: any[]) => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => fn(...args), delay);
  };
};

function syncLabelPosition(rect: BoxRect) {
  if (!rect.labelObj || !rect.labelBg) return;
  const txt = rect.labelObj;
  const bg = rect.labelBg;
  txt.set({
    left: (rect.left ?? 0) + 4,
    top: (rect.top ?? 0) - 5,
  });
  txt.setCoords();
  bg.set({
    left: (txt.left ?? 0) - 2,
    top: (txt.top ?? 0) - 2,
    width: (txt.width ?? 0) + 4,
    height: (txt.height ?? 0) + 1,
  });
  bg.setCoords();
}

const onContainerResize = debounce(() => {
  const cv = canvas;
  const img = bgImage;
  const container = containerRef.value;
  if (!cv || !img || !container) return;

  const cw = container.clientWidth - 20;
  const ch = container.clientHeight - 20;
  if (cw < 10 || ch < 10) return;
  cv.setDimensions({ width: cw, height: ch });

  const originW = img.width;
  const originH = img.height;
  originScale = Math.min(cw / originW, ch / originH, 1);
  const dispW = originW * originScale;
  const dispH = originH * originScale;

  img.set({
    scaleX: originScale,
    scaleY: originScale,
    left: (cw - dispW) / 2,
    top: (ch - dispH) / 2,
  });

  const rects = getBoxRects(cv);
  rects.forEach((rect) => {
    const bd = rect.boxData;
    if (!bd) return;
    const imgLeft = (img.left as number) ?? 0;
    const imgTop = (img.top as number) ?? 0;
    const rx1 = bd.x1 * originScale;
    const ry1 = bd.y1 * originScale;
    const rx2 = bd.x2 * originScale;
    const ry2 = bd.y2 * originScale;
    rect.set({
      left: imgLeft + rx1,
      top: imgTop + ry1,
      width: rx2 - rx1,
      height: ry2 - ry1,
    });
    rect.setCoords();
    syncLabelPosition(rect);
  });
  cv.renderAll();
});
const EXCLUDED_BOX_FIELDS = ["drawing_type"];
const loadImageAndBox = async () => {
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
    const cv = canvas;
    cv.clear();
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
    const cw = (containerDom?.clientWidth ?? 720) - 20;
    const ch = (containerDom?.clientHeight ?? 520) - 20;
    originScale = Math.min(cw / originW, ch / originH, 1);
    const dispW = originW * originScale;
    const dispH = originH * originScale;
    cv.setDimensions({ width: cw, height: ch });
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
    cv.add(img);
    const boxItems: [string, DetecBoxItem][] = Object.entries(
      props.detecBoxs || {}
    ).filter(([fieldKey]) => !EXCLUDED_BOX_FIELDS.includes(fieldKey));
    boxItems.forEach(([, item]) => {
      if (!item?.tokens_bbox) return;
      const { x0, y0, x1, y1 } = item.tokens_bbox;
      const conf = item.confidence ?? 1;
      if (conf !== undefined && conf < props.confThreshold) return;
      const normalized: NormalizedBox = {
        ...item,
        x1: x0,
        y1: y0,
        x2: x1,
        y2: y1,
        label: item.value,
        conf: conf,
      };
      if (normalized.x1 === normalized.x2 || normalized.y1 === normalized.y2)
        return;

      const rx1 = normalized.x1 * originScale;
      const ry1 = normalized.y1 * originScale;
      const rx2 = normalized.x2 * originScale;
      const ry2 = normalized.y2 * originScale;

      const rect = new Rect({
        originY: "top",
        originX: "left",
        left: (img.left as number) + rx1,
        top: (img.top as number) + ry1 + 0.2,
        width: rx2 - rx1,
        height: ry2 - ry1,
        stroke: "#f53f3f",
        strokeWidth: 0.5,
        fill: "transparent",
        strokeUniform: true,
        selectable: true,
        evented: true,
        lockMovementX: true,
        lockMovementY: true,
        lockScalingX: true,
        lockScalingY: true,
        hasControls: false,
        hasBorders: false,
      }) as unknown as BoxRect;

      rect.boxData = normalized;
      rect.setCoords();
      cv.add(rect);
    });
    cv.renderAll();
  } catch (err) {

    loadError.value = true;
  } finally {
    isLoading.value = false;
    loadingLock = false;
  }
};

const initCanvas = async () => {
  await nextTick();
  if (!fabricCanvasRef.value || !containerRef.value) await nextTick();
  if (!fabricCanvasRef.value || !containerRef.value) {
    return;
  }
  const cw = containerRef.value.clientWidth - 20 || 700;
  const ch = containerRef.value.clientHeight - 20 || 500;

  const cv = new Canvas(fabricCanvasRef.value, {
    width: cw,
    height: ch,
    selection: false,
    preserveObjectStacking: true,
  });
  canvas = cv;
  bindCanvasEvent(cv);
  resizeObserver = new ResizeObserver(onContainerResize);
  resizeObserver.observe(containerRef.value);
};

const bindCanvasEvent = (cv: Canvas) => {
  let isLeftDrag = false;
  let isPanning = false;
  let lastPosX = 0;
  let lastPosY = 0;

  cv.on("mouse:wheel", (opt) => {
    const evt = opt.e as WheelEvent;
    const delta = evt.deltaY;
    let zoom = cv.getZoom();
    zoom *= 0.999 ** delta;
    zoom = Math.max(0.2, Math.min(5, zoom));
    cv.zoomToPoint(new Point(evt.offsetX, evt.offsetY), zoom);
    evt.preventDefault();
    evt.stopPropagation();
  });

  cv.on("mouse:down", (opt) => {
    const evt = opt.e as MouseEvent;

    if (evt.button === 1) {
      isPanning = true;
      lastPosX = evt.clientX;
      lastPosY = evt.clientY;
    }

    if (evt.button === 0 && !opt.target) {
      isLeftDrag = true;
      lastPosX = evt.clientX;
      lastPosY = evt.clientY;
    }
  });

  cv.on("mouse:move", (opt) => {
    const evt = opt.e as MouseEvent;

    if (isPanning || isLeftDrag) {
      const dx = evt.clientX - lastPosX;
      const dy = evt.clientY - lastPosY;
      cv.relativePan(new Point(dx, dy));
      lastPosX = evt.clientX;
      lastPosY = evt.clientY;
    }
  });

  cv.on("mouse:up", () => {
    isPanning = false;
    isLeftDrag = false;
  });

  cv.on("object:removed", (e) => {
    const obj = e.target as BoxRect | undefined;
    if (!obj) return;
    if (obj.labelObj) cv.remove(obj.labelObj);
    if (obj.labelBg) cv.remove(obj.labelBg);
  });
};

const fieldNameMap: Record<string, string> = {
  drawing_number: "Drawing number",
  project_name: "Name of project",
  drawing_title: "Name of drawing",
  architect: "Architect",
  draughtsperson: "Draughtsperson",
  date: "Date",
  scale: "Scale",
  drawing_type: "Drawing type",
};

const activeFieldKey = ref<string | null>(null);

const detecBoxEntries = computed<DetecBoxEntry[]>(() => {
  if (!props.detecBoxs) return [];
  return Object.entries(props.detecBoxs).map(([k, v]) => ({
    fieldKey: k,
    ...v,
  }));
});

const formatConfidence = (conf?: number) =>
  `${((conf ?? 0) * 100).toFixed(1)}%`;

const getFieldValue = (fieldKey: string): string => {
  const val = localDetecBoxs.value[fieldKey]?.value;
  return val === null || val === undefined ? "" : String(val);
};

const setFieldValue = (fieldKey: string, val: string | number) => {
  const item = localDetecBoxs.value[fieldKey];
  if (!item) return;
  item.value = val;
};

const handleClickItem = (fieldKey: string, item: DetecBoxEntry) => {
  if (EXCLUDED_BOX_FIELDS.includes(fieldKey)) {
    return 0;
  }
  if (activeFieldKey.value === fieldKey) {
    resetAllBoxHighlight();
    activeFieldKey.value = null;
    return;
  }
  activeFieldKey.value = fieldKey;
  resetAllBoxHighlight();

  if (!item.tokens_bbox) return;

  const { x0, y0, x1, y1 } = item.tokens_bbox;
  const cv = canvas;
  if (!cv) return;

  const targetRect = getBoxRects(cv).find((rect) => {
    const bd = rect.boxData;
    if (!bd || !bd.tokens_bbox) return false;
    return (
      bd.tokens_bbox.x0 === x0 &&
      bd.tokens_bbox.y0 === y0 &&
      bd.tokens_bbox.x1 === x1 &&
      bd.tokens_bbox.y1 === y1
    );
  });

  if (targetRect) {
    targetRect._originStroke = targetRect.stroke as string;
    targetRect._originStrokeWidth = targetRect.strokeWidth;
    targetRect.set({
      stroke: "#ffc107",
      strokeWidth: 1,
    });
    cv.renderAll();
  }
};

const resetAllBoxHighlight = () => {
  const cv = canvas;
  if (!cv) return;
  const rects = getBoxRects(cv);
  rects.forEach((rect) => {
    if (rect._originStroke !== undefined) {
      rect.set({
        stroke: rect._originStroke,
        strokeWidth: rect._originStrokeWidth,
      });
      delete rect._originStroke;
      delete rect._originStrokeWidth;
    }
  });
  cv.renderAll();
};

const submit = () => {
  ElMessageBox.confirm("Whether to confirm the audit?", "Warning", {
    confirmButtonText: "Confirmation",
    cancelButtonText: "cancel",
    type: "warning",
  })
    .then(async () => {
      axios
        .post(Api.manualReview, {
          image_key: props.image_key,
          result: formatSubmitPayload(unref(localDetecBoxs)).result,
        })
        .then(function (response) {
          ElMessage({
            message: response.data.message,
            type: "success",
          });
          emit("saved", response.data.result.image_key);
        })
        .catch(function (error) {
          const isNetworkErr = error.message === "Network Error";
          if (isNetworkErr) {
            ElMessage({
              message:
                "The backend service connection failed. Please check if the service is started and if the network is connected.",
              type: "error",
            });
          } else {
            ElMessage({
              message:
                "Query of task status failed:" +
                (error.message || "Unknown error"),
              type: "error",
            });
          }
        });
    })
    .catch(() => {});
};

const formatSubmitPayload = (localBoxObj: DetecBoxMap) => {
  const submitResult: Record<string, string | number | null> = {};
  Object.entries(localBoxObj).forEach(([fieldKey, itemObj]) => {
    submitResult[fieldKey] = itemObj?.value ?? null;
  });
  return { result: submitResult };
};

watch(
  () => [props.urlImage, props.thumbUrlImage],
  () => loadImageAndBox(),
  { deep: true }
);

watch(
  () => props.detecBoxs,
  (val) => {
    localDetecBoxs.value = JSON.parse(JSON.stringify(val || {})) as DetecBoxMap;
    activeFieldKey.value = null;
    resetAllBoxHighlight();
    loadImageAndBox();
  },
  { deep: true, immediate: true }
);

defineExpose({
  triggerDraw: loadImageAndBox,
});

onUnmounted(() => {
  if (resizeTimer) clearTimeout(resizeTimer);
  if (resizeObserver) resizeObserver.disconnect();
  canvas?.dispose();
});
</script>

<style scoped lang="less">
.compare-root {
  display: grid;
  grid-template-columns: 3fr 1fr;
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
.left-col,
.right-col {
  overflow: hidden;
}
.col-label {
  padding: 8px 12px;
  background: #f5f7fa;
  font-weight: 500;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
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
.main-img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
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

.list-message {
  overflow-y: auto;
  padding: 8px;
  height: calc(60vh - 40px);
  .item {
    border: 1px solid #e4e7ed;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .item:hover {
    background-color: #f5f7fa;
  }
  .item.active {
    background-color: #e6f7ff;
    border-color: #409eff;
  }
  .field-name {
    font-weight: 600;
    font-size: 13px;
    color: #303133;
    margin-bottom: 4px;
  }
  .field-value {
    font-size: 12px;
    color: #606266;
    word-break: break-all;
    margin-bottom: 6px;
    max-height: 60px;
    overflow: hidden;
  }
  .field-meta {
    font-size: 11px;
    flex-wrap: wrap;
    span {
      display: block;
    }
  }
  .conf {
    color: #909399;
  }
  .status-tag {
    padding: 1px 6px;
    border-radius: 3px;
  }
  .tag-accept {
    background: rgba(103, 194, 58, 0.15);
    color: #67c23a;
  }
  .tag-review {
    background: rgba(230, 162, 60, 0.15);
    color: #e6a23c;
  }
  .tag-reviewed {
    background: rgba(64, 158, 255, 0.15);
    color: #409eff;
  }
  .tag-red {
    background: rgba(230, 162, 60, 0.15);
    color: #e41515;
  }
  .empty-item {
    text-align: center;
    padding: 24px;
    color: #909399;
  }
}
</style>
