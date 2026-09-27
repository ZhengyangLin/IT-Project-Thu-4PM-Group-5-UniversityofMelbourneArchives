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
          v-for="(item, fieldKey) in detecBoxEntries"
          :key="fieldKey"
          :class="{ active: activeFieldKey === fieldKey }"
          @click="handleClickItem(fieldKey, item)"
          v-show="fieldNameMap[item.fieldKey] != 'Drawing type'"
        >
          <!-- {{item}} -->
          <!-- <div class="field-name">{{ fieldKey }}</div> -->
          <div class="field-name">
            {{ fieldNameMap[item.fieldKey] || fieldKey }}
          </div>
          <div class="field-value" :title="item.value">
            <span v-if="!isEdit"> Final value:{{ item.value ?? "-" }}</span>
            
            <span v-else>
              Final value:
              <el-input
                v-model="localDetecBoxs[item.fieldKey].value"
                placeholder=""
                size="small"
              />
            </span>
          </div>
          <div class="field-meta">
            <div class="conf">
              Confidence:{{ (item.confidence * 100).toFixed(1) }}%
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
              }"
            >
              {{ item.review_status }}
            </span>
          </div>
        </div>
        <div v-if="detecBoxEntries.length === 0" class="empty-item">
          No labeled data available.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang='ts'>
import { ref, watch, nextTick, onUnmounted, computed, unref } from "vue";
import { Warning } from "@element-plus/icons-vue";
import { Canvas, FabricImage, Rect, FabricText } from "fabric";
import { ElMessageBox, ElMessage } from "element-plus";
import axios from "axios";
const props = defineProps({
  urlImage: String,
  thumbUrlImage: { type: String, default: "" },
  detecBoxs: { type: Object, default: () => {} },
  image_key: { type: String, default: "" },
  confThreshold: { type: Number, default: 0.3 },
  isEdit: {
    type: Boolean,
    default: false,
  },
});
enum Api {
  manualReview = "http://127.0.0.1:8000/Unimelb/ocr-results/manual-review",
}
const emit = defineEmits(["box-change", "fabricImgChange"]);
const fabricCanvasRef = ref(null);
const containerRef = ref(null);
let canvas = null;
let bgImage = null;
let originScale = 1;
let resizeObserver = null;
let resizeTimer = null;
const mode = ref("view");
let drawStart = null;
let tempRect = null;
let loadingLock = false;
const isLoading = ref(false);
const loadError = ref(false);

const localDetecBoxs = ref({});
const getDrawUrl = () => props.thumbUrlImage || props.urlImage;
console.log(getDrawUrl);

const debounce = (fn, delay = 120) => {
  return (...args) => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => fn(...args), delay);
  };
};

function createBoxLabel(textStr, rect) {
  const labelText = new FabricText(textStr, {
    left: rect.left + 0,
    top: rect.top + 10,
    fontSize: 5,
    fill: "#f53f3f",
    originX: "left",
    originY: "top",
    selectable: false,
    evented: false,
  });
  canvas.add(labelText);
  const textW = labelText.width;
  const textH = labelText.height;
  canvas.remove(labelText);
  const labelBg = new Rect({
    originX: "left",
    originY: "top",
    width: textW,
    height: textH,
    fill: "rgba(255, 255, 255, 0.75)",
    selectable: false,
    evented: false,
    //  width: 100, height: 100, fill: 'orange',
  });
  rect.labelObj = labelText;
  rect.labelBg = labelBg;
  return { labelText, labelBg };
}

function syncLabelPosition(rect) {
  if (!rect.labelObj || !rect.labelBg) return;
  const txt = rect.labelObj;
  const bg = rect.labelBg;
  txt.set({
    left: rect.left + 4,
    top: rect.top - 5,
  });
  txt.setCoords();
  bg.set({
    left: txt.left - 2,
    top: txt.top - 2,
    width: txt.width + 4,
    height: txt.height + 1,
  });
  bg.setCoords();
}

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
  console.log(
    containerRef.value.clientWidth,
    containerRef.value.clientHeight,
    originW,
    originH,
    dispW
  );
  bgImage.set({
    scaleX: originScale,
    scaleY: originScale,
    left: (cw - dispW) / 2,
    top: (ch - dispH) / 2,
  });

  const rects = canvas.getObjects("rect").filter((o) => o.boxData);
  rects.forEach((rect) => {
    const { x1, y1, x2, y2 } = rect.boxData;
    const rx1 = x1 * originScale;
    const ry1 = y1 * originScale;
    const rx2 = x2 * originScale;
    const ry2 = y2 * originScale;
    rect.set({
      left: bgImage.left + rx1,
      top: bgImage.top + ry1,
      width: rx2 - rx1,
      height: ry2 - ry1,
    });
    rect.setCoords();
    syncLabelPosition(rect);
  });
  canvas.renderAll();
});

const normalizeBox = (box) => {
  let x1, y1, x2, y2;
  if (
    box.x1 !== undefined &&
    box.y1 !== undefined &&
    box.x2 !== undefined &&
    box.y2 !== undefined
  ) {
    x1 = box.x1;
    y1 = box.y1;
    x2 = box.x2;
    y2 = box.y2;
  } else if (
    box.x !== undefined &&
    box.y !== undefined &&
    box.w !== undefined &&
    box.h !== undefined
  ) {
    x1 = box.x;
    y1 = box.y;
    x2 = box.x + box.w;
    y2 = box.y + box.h;
  } else if (
    box.left !== undefined &&
    box.top !== undefined &&
    box.width !== undefined &&
    box.height !== undefined
  ) {
    x1 = box.left;
    y1 = box.top;
    x2 = box.left + box.width;
    y2 = box.top + box.height;
  } else {
    return null;
  }
  if (x1 > x2) [x1, x2] = [x2, x1];
  if (y1 > y2) [y1, y2] = [y2, y1];
  return { ...box, x1, y1, x2, y2 };
};

const loadImageAndBox = async () => {
  const url = getDrawUrl();
  console.log(url);
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
    // console.log(bgImage);
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

    const boxItems = Object.values(props.detecBoxs || {});
    boxItems.forEach((item) => {
      //
      if (!item?.tokens_bbox) return;
      const { x0, y0, x1, y1 } = item.tokens_bbox;
      //
      const conf = item.confidence ?? 1;
      if (conf !== undefined && conf < props.confThreshold) return;

      //
      const normalized = {
        ...item,
        x1: x0,
        y1: y0,
        x2: x1,
        y2: y1,
        label: item.value, //
        conf: conf,
      };

      //
      if (normalized.x1 === normalized.x2 || normalized.y1 === normalized.y2)
        return;

      const rx1 = normalized.x1 * originScale;
      const ry1 = normalized.y1 * originScale;
      const rx2 = normalized.x2 * originScale;
      const ry2 = normalized.y2 * originScale;

      const rect = new Rect({
        originY: "top",
        originX: "left",
        left: img.left + rx1,
        top: img.top + ry1 + 0.2,
        width: rx2 - rx1,
        height: ry2 - ry1,
        stroke: "#f53f3f",
        strokeWidth: 0.5,
        fill: "transparent",
        strokeUniform: true,
        selectable: true,
        evented: true,
        //
        lockMovementX: true,
        lockMovementY: true,
        lockScalingX: true,
        lockScalingY: true,
        hasControls: false, //
        hasBorders: false, //
      });
      rect.boxData = normalized;
      rect.setCoords();
      canvas.add(rect);

      if (normalized.label != null) {
      }
    });
    canvas.renderAll();
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

const bindCanvasEvent = () => {
  let isLeftDrag = false;
  canvas.on("mouse:wheel", (opt) => {
    const delta = opt.e.deltaY;
    let zoom = canvas.getZoom();
    zoom *= 0.999 ** delta;
    zoom = Math.max(0.2, Math.min(5, zoom));
    canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
    opt.e.preventDefault();
    opt.e.stopPropagation();
  });

  canvas.on("mouse:down", (opt) => {
    const evt = opt.e;

    if (evt.button === 1) {
      canvas.isDragging = true;
      canvas.lastPosX = evt.clientX;
      canvas.lastPosY = evt.clientY;
    }

    if (evt.button === 0 && !opt.target) {
      isLeftDrag = true;
      canvas.lastPosX = evt.clientX;
      canvas.lastPosY = evt.clientY;
    }
  });

  canvas.on("mouse:move", (opt) => {
    const evt = opt.e;

    if (canvas.isDragging) {
      const dx = evt.clientX - canvas.lastPosX;
      const dy = evt.clientY - canvas.lastPosY;
      canvas.relativePan({ x: dx, y: dy });
      canvas.lastPosX = evt.clientX;
      canvas.lastPosY = evt.clientY;
    }
    if (isLeftDrag) {
      const dx = evt.clientX - canvas.lastPosX;
      const dy = evt.clientY - canvas.lastPosY;
      canvas.relativePan({ x: dx, y: dy });
      canvas.lastPosX = evt.clientX;
      canvas.lastPosY = evt.clientY;
    }
  });

  canvas.on("mouse:up", () => {
    canvas.isDragging = false;
    isLeftDrag = false;
  });

  canvas.on("object:removed", (e) => {
    const obj = e.target;
    if (obj.labelObj) canvas.remove(obj.labelObj);
    if (obj.labelBg) canvas.remove(obj.labelBg);
  });
};
const getBoxList = () => {
  return canvas
    .getObjects("rect")
    .filter((item) => item.boxData)
    .map((item) => item.boxData);
};

const handleGetAllBox = () => {
  const list = getBoxList();

  alert(JSON.stringify(list, null, 2));
};

const resetZoom = () => {
  if (!canvas) return;
  canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
};

const clearSelect = () => {
  const active = canvas.getActiveObject();
  if (active) {
    canvas.remove(active);
    emit("box-change", getBoxList());
  }
};
const fieldNameMap = {
  drawing_number: "Drawing number",
  project_name: "Name of project",
  drawing_title: "Name of drawing",
  architect: "Architect",
  draughtsperson: "Draughtsperson",
  date: "Date",
  scale: "Scale",
  drawing_type: "Drawing type",
};
const activeFieldKey = ref(null);

const detecBoxEntries = computed(() => {
  if (!props.detecBoxs) return [];
  return Object.entries(props.detecBoxs).map(([k, v]) => ({
    fieldKey: k,
    ...v,
  }));
});

const handleClickItem = (fieldKey, item) => {
  if (activeFieldKey.value === fieldKey) {
    resetAllBoxHighlight();
    activeFieldKey.value = null;
    return;
  }
  activeFieldKey.value = fieldKey;
  resetAllBoxHighlight();

  if (!item.tokens_bbox) return;

  const { x0, y0, x1, y1 } = item.tokens_bbox;

  const targetRect = canvas?.getObjects("rect")?.find((rect) => {
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
    targetRect._originStroke = targetRect.stroke;
    targetRect._originStrokeWidth = targetRect.strokeWidth;
    targetRect.set({
      stroke: "#ffc107",
      strokeWidth: 1,
    });
    canvas.renderAll();
  }
};
const resetAllBoxHighlight = () => {
  if (!canvas) return;
  const rects = canvas.getObjects("rect").filter((o) => o.boxData);
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
  canvas.renderAll();
};
const submit = () => {
  console.log(props.image_key, localDetecBoxs.value);
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
          emit("fabricImgChange", 0);
        })
        .catch(function (error) {});
    })
    .catch(() => {});
};

const formatSubmitPayload = (localBoxObj) => {
  const submitResult = {};
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
    localDetecBoxs.value = JSON.parse(JSON.stringify(val || {}));
    console.log(unref(localDetecBoxs));
    activeFieldKey.value = null;
    resetAllBoxHighlight();
    loadImageAndBox();
  },
  { deep: true, immediate: true }
);


defineExpose({
  triggerDraw: loadImageAndBox,
  getAllBoxes: getBoxList,
});

onUnmounted(() => {
  clearTimeout(resizeTimer);
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
  // flex: 1;
  overflow-y: auto;
  padding: 8px;
  height: calc(60vh - 40px);
  overflow-y: auto;
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
    // display: flex;
    // justify-content: space-between;
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
  .empty-item {
    text-align: center;
    padding: 24px;
    color: #909399;
  }
}
</style>
