<template>
  <div class="image-task-page">

    <div class="bg-layer"></div>


    <div class="page-header">
      <h1 class="page-title">
        <el-icon><DataAnalysis /></el-icon>
        Image Task Monitoring Center
      </h1>
      <div class="page-subtitle">
        Real-time viewing of image directories, task statuses and detection
        results
      </div>
    </div>

 
    <div class="stat-overview">
      <div
        class="stat-card stat-card-total"
        :class="statusValue == null ? 'on' : ''"
        @click="tabStatus(null)"
      >
        <div class="stat-icon">
          <el-icon><Picture /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">All the pictures</div>
          <div class="stat-value">{{ pageData.total }}</div>
        </div>
      </div>
      <div
        class="stat-card stat-card-wait"
        :class="statusValue == 0 ? 'on' : ''"
        @click="tabStatus(0)"
      >
        <div class="stat-icon">
          <el-icon><Clock /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">To be tested</div>
          <div class="stat-value">{{ pageData.status_counts[0] }}</div>
        </div>
      </div>

      <div
        class="stat-card stat-card-running"
        :class="statusValue == 1 ? 'on' : ''"
        @click="tabStatus(1)"
      >
        <div class="stat-icon">
          <el-icon><Loading /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">In the process of testing</div>
          <div class="stat-value">{{ pageData.status_counts[1] }}</div>
        </div>
      </div>
      <div
        class="stat-card stat-card-success"
        :class="statusValue == 2 ? 'on' : ''"
        @click="tabStatus(2)"
      >
        <div class="stat-icon">
          <el-icon><CircleCheck /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">Successfully</div>
          <div class="stat-value">{{ pageData.status_counts[2] }}</div>
        </div>
      </div>
      <div
        class="stat-card stat-card-fail"
        :class="statusValue == 3 ? 'on' : ''"
        @click="tabStatus(3)"
      >
        <div class="stat-icon">
          <el-icon><CircleClose /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-label">Failure</div>
          <div class="stat-value">{{ pageData.status_counts[3] }}</div>
        </div>
      </div>
    </div>

 
    <div class="main-grid">
 
      <div class="panel-card folder-panel">
        <div class="panel-title" @click="clearFilter">
          <el-icon><Folder /></el-icon>
          Folder statistics
        </div>
        <div class="folder-list-wrap">
          <div class="folder-item header-row">
            <div class="col-name">Folder name</div>
            <div class="col-count">Images Number</div>
          </div>
          <div
            class="folder-item"
            v-for="item in folderList"
            :key="item.folderName"
            @click="openFolder(item.folderName)"
            :class="item.folderName == currentFile ? 'file-on' : ''"
          >
            <div class="col-name">{{ item.folderName }}</div>
            <div class="col-count">{{ item.count }}</div>
          </div>
        </div>
      </div>


      <div class="panel-card image-panel">
        <div class="panel-title">
          <el-icon><Document /></el-icon>
          Image task list

          <el-button
            class="btn"
            type="primary"
            style="margin-left: auto"
            :icon="Refresh"
            @click="submitRecheck(1)"
            >Batch recheck</el-button
          >
        </div>

 
        <div class="img-list-wrap">
          <div class="img-row header-row">
            <div class="col-select">select</div>
            <div class="col-preview">preview</div>
            <div class="col-folder">Folder</div>
            <div class="col-nodeid">node_id</div>
            <div class="col-imgname">Image name</div>
            <div class="col-status">Status</div>
            <div class="col-status">Review status</div>
       
            <div class="col-error">Error message</div>
            <div class="col-update">Operation</div>
          </div>
          <div class="img-box-list" v-if="pageImageList?.length">
    
            <div class="img-row" v-for="row in pageImageList" :key="row.path">
              <div class="col-select">
                <el-checkbox
                  :model-value="imgKeyList.includes(row.image_key)"
                  @click="checkRow(row)"
                  label=""
                  size="small"
                />
              </div>
              <div class="col-preview">
                <img :src="row.image_url" alt="" />
              </div>
              <div class="col-folder">{{ row.folder }}</div>
              <div class="col-nodeid">{{ row.node_id }}</div>
              <div class="col-imgname" title="{{ row.image_name }}">
                {{ row.image_name }}
              </div>
              <div class="col-status">
                <span class="status-tag" :class="getStatusTag(row.status).cls">
                  {{ getStatusTag(row.status).text }}
                </span>
              </div>
              <div class="col-status">
                <span class="status-tag" v-if="row.status != 2"> </span>
                <span
                  v-else
                  class="status-tag"
                  :class="getStatusTag(row.manual_reviewed == 0 ? 3 : 2).cls"
                >
                  {{ row.manual_reviewed == 0 ? "Pending review" : "Reviewed" }}
                </span>
              </div>
              <div class="col-error text-fail">
                <el-tooltip
                  class="box-item"
                  effect="dark"
                  :content="row.error"
                  placement="top"
                >
                  <span>{{ row.error || "-" }}</span>
                </el-tooltip>
              </div>
              <div class="col-update">
                <el-button
                  class="btn"
                  type="primary"
                  size="small"
                  v-show="row.status == 2"
                  @click="openResult(row)"
                  :icon="View"
                  >Test results</el-button
                >
                <el-button
                  class="btn"
                  type="success"
                  size="small"
                  v-show="row.status == 2"
                  @click="editResult(row)"
                  :icon="Edit"
                  >Manual review</el-button
                >
                <el-button
                  class="btn"
                  type="warning"
                  size="small"
                  :icon="Aim"
                  @click="recheck(row)"
                  >Recheck</el-button
                >
              </div>
            </div>
          </div>


          <div v-if="!pageImageList?.length" class="empty-tip">
            No image task data available at present.
          </div>
       
        </div>
      </div>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, onUnmounted, unref, nextTick } from "vue";
import {
  DataAnalysis,
  Picture,
  Clock,
  List,
  Loading,
  CircleCheck,
  CircleClose,
  Folder,
  Document,
  Edit,
  View,
  Aim,
  Refresh,
} from "@element-plus/icons-vue";
import { ElMessageBox, ElMessage } from "element-plus";
import axios from "axios";
const loading = ref(false);

const pageData = ref({
  folders: {},
  total: 0,
  status_counts: {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
  },
  images: [],
});

const folderList = computed(() => {
  const folders = pageData.value.folders || {};
  return Object.entries(folders).map(([folderName, count]) => ({
    folderName,
    count,
  }));
});

const getStatusTag = (status) => {
  const map = {
    0: { text: "To be tested", cls: "tag-wait" },

    1: { text: "In the process of testing", cls: "tag-running" },
    2: { text: "Completed", cls: "tag-success" },
    3: { text: "Failure", cls: "tag-fail" },
  };
  return map[status] || { text: "Unknown", cls: "tag-unknown" };
};


</script>
<style scoped lang="less">
.image-task-page {
  position: relative;
  min-height: 100%;
  // padding: 24px;
  color: #fff;
  font-size: 14px;
  background-image: url("@/assets/images/bg.jpeg");
  background-size: 100% 100%;
}

.bg-layer {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -2;
  background: linear-gradient(
      to bottom,
      rgba(10, 14, 28, 0.96),
      rgba(10, 14, 28, 0.88)
    ),
    url("@/assets/bg_dark_tech_01.png") center/cover no-repeat;
}
.page-header {
  text-align: center;
  margin-bottom: 24px;
}

.page-title {
  margin: 0 0 8px;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 1px;
  background: linear-gradient(135deg, #00d4ff, #7b61ff);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.page-subtitle {
  color: #aeb7c2;
  font-size: 14px;
}

.stat-overview {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  margin-bottom: 24px;
  .stat-card {
    cursor: pointer;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    display: flex;
    align-items: center;
    gap: 14px;
    transition: 0.3s;
    .stat-label {
      color: #aeb7c2;
      font-size: 13px;
      margin-bottom: 4px;
    }
  }
  .on {
    transform: translateY(-4px);
    border-color: rgba(0, 212, 255, 0.35);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    background: linear-gradient(
      90deg,
      rgba(0, 212, 255, 0.25),
      rgba(0, 150, 168, 0.1)
    );
    border: 2px solid rgba(0, 212, 255, 0.35);
    color: #fff;
    .stat-label {
      color: #fff;
      font-size: 1rem;
    }
  }
}

.stat-card:hover {
  transform: translateY(-4px);
  border-color: rgba(0, 212, 255, 0.35);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.stat-icon {
  width: 46px;
  height: 46px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: #fff;
  flex-shrink: 0;
}

.stat-card-total .stat-icon {
  background: linear-gradient(135deg, #00d4ff, #7b61ff);
}
.stat-card-wait .stat-icon {
  background: linear-gradient(135deg, #909399, #606266);
}
.stat-card-queue .stat-icon {
  background: linear-gradient(135deg, #e6a23c, #f5d76e);
}
.stat-card-running .stat-icon {
  background: linear-gradient(135deg, #409eff, #79bbff);
}
.stat-card-success .stat-icon {
  background: linear-gradient(135deg, #67c23a, #95d475);
}
.stat-card-fail .stat-icon {
  background: linear-gradient(135deg, #f56c6c, #fab6b6);
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: 800;
}
.stat-card-total .stat-value {
  color: #00d4ff;
}
.stat-card-wait .stat-value {
  color: #c9cdcf;
}
.stat-card-queue .stat-value {
  color: #f5d76e;
}
.stat-card-running .stat-value {
  color: #79bbff;
}
.stat-card-success .stat-value {
  color: #95d475;
}
.stat-card-fail .stat-value {
  color: #fab6b6;
}

.main-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 20px;
}

.panel-card {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 16px;
  padding: 20px;
  backdrop-filter: blur(12px);
  overflow: hidden;
}
.panel-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #fff;
}
.folder-panel {
  height: fit-content;
}
.image-panel {
  min-height: 420px;
}


.folder-list-wrap {
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  .folder-item {
    cursor: pointer;
    display: flex;
    padding: 12px 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
  .file-on {
    background: #1890ff;
  }
}

.folder-item:last-child {
  border-bottom: none;
}
.folder-item.header-row {
  background: rgba(255, 255, 255, 0.08);
  font-weight: bold;
}
.col-name {
  flex: 1;
}
.col-count {
  width: 80px;
  text-align: right;
}


.img-list-wrap {
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  .pagination-wrap {
    padding: 14px 10px;
    display: flex;
    justify-content: flex-end;
    :deep(.el-pagination) {
      --el-pagination-bg-color: rgba(255, 255, 255, 0.06);
      --el-pagination-text-color: #fff;
      --el-pagination-hover-color: #00d4ff;
    }
  }
}
.img-box-list {
  height: calc(100vh - 400px);
  overflow-y: auto;
}
.img-row {
  display: flex;
  align-items: center;
  padding: 12px 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  gap: 8px;
}
.img-row:last-child {
  border-bottom: none;
}
.img-row.header-row {
  background: rgba(255, 255, 255, 0.08);
  font-weight: bold;
}
.img-row:hover {
  background: rgba(255, 255, 255, 0.04);
}
.col-select {
  width: 50px;
  word-wrap: break-word;
}
.col-preview {
  width: 100px;
  flex-shrink: 0;
  word-wrap: break-word;
}
.col-folder {
  width: 140px;
  flex-shrink: 0;
  word-wrap: break-word;
}
.col-nodeid {
  width: 100px;
  flex-shrink: 0;
  word-wrap: break-word;
}
.col-imgname {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.col-status {
  width: 110px;
  flex-shrink: 0;
  word-wrap: break-word;
}
.col-uuid {
  width: 200px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  word-wrap: break-word;
}
.col-error {
  width: 250px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  // white-space: nowrap;
  word-wrap: break-word;
}
.col-update {
  width: 140px;
  flex-shrink: 0;
  // display: flex;
  // flex-wrap: wrap;
  .btn {
    display: block;
    margin: 0;
    margin-bottom: 0.5rem;
  }
}


.status-tag {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 12px;
}
.tag-wait {
  background: rgba(144, 147, 153, 0.25);
  color: #c9cdcf;
}
.tag-queue {
  background: rgba(230, 162, 60, 0.25);
  color: #f5d76e;
}
.tag-running {
  background: rgba(64, 158, 255, 0.25);
  color: #79bbff;
}
.tag-success {
  background: rgba(103, 194, 58, 0.25);
  color: #95d475;
}
.tag-fail {
  background: rgba(245, 108, 108, 0.25);
  color: #fab6b6;
}
.tag-unknown {
  background: rgba(255, 255, 255, 0.12);
  color: #fff;
}

.text-fail {
  color: #f56c6c;
}
.empty-tip {
  padding: 40px;
  text-align: center;
  color: #aeb7c2;
}


@media (max-width: 1400px) {
  .stat-overview {
    grid-template-columns: repeat(3, 1fr);
  }
  .main-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 768px) {
  .image-task-page {
    padding: 16px;
  }
  .page-title {
    font-size: 22px;
  }
  .stat-overview {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
