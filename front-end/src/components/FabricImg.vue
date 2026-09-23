<template>
  <div class="compare-root">
    <div class="compare-col left-col">
      <div class="col-label">Annotation diagram</div>
      <div class="scroll-wrap canvas-wrap" ref="containerRef">
        <el-image
          ref="elImgRef"
          :src="getDrawUrl()"
          fit="contain"
          class="preview-image"
          @load="onImageLoad"
          @error="onImageError"
        />

        <div v-if="isLoading" class="canvas-loading">
          <div class="loading-text">Please wait while the picture loads...</div>
        </div>

        <div v-if="loadError && !isLoading" class="canvas-error">
          <el-icon color="red"><Warning /></el-icon>
          <span>Picture Load Failed</span>
          <el-button size="small" @click="reloadImage" style="margin-left:8px">Retry</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, onUnmounted } from "vue";
import { Warning } from "@element-plus/icons-vue";

const props = defineProps({
  urlImage: String,
  thumbUrlImage: { type: String, default: "" },
});

const elImgRef = ref(null);
const containerRef = ref(null);

const isLoading = ref(true);
const loadError = ref(false);

const getDrawUrl = () => props.thumbUrlImage || props.urlImage;

const onImageLoad = () => {
  isLoading.value = false;
  loadError.value = false;
};

const onImageError = () => {
  isLoading.value = false;
  loadError.value = true;
};

const reloadImage = () => {
  isLoading.value = true;
  loadError.value = false;
  const url = getDrawUrl();
  if (!url) {
    isLoading.value = false;
    loadError.value = true;
    return;
  }
  (elImgRef.value as any)?.setSrc(`${url}?_t=${Date.now()}`);
};

watch(
  () => [props.urlImage, props.thumbUrlImage],
  () => {
    if (!getDrawUrl()) {
      isLoading.value = false;
      loadError.value = true;
      return;
    }
    isLoading.value = true;
    loadError.value = false;
  },
  { deep: true }
);

defineExpose({
  triggerDraw: reloadImage,
});

onUnmounted(() => {
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

  .preview-image {
    // max-width: 100%;
    height: 100%;
  }
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
