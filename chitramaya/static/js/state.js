// ── DOM References ────────────────────────────────────────
const appRoot = document.getElementById('appRoot');

// Header
const configBtn = document.getElementById('configBtn');
const configMenu = document.getElementById('configMenu');
const cfgSave = document.getElementById('cfgSave');
const cfgLoad = document.getElementById('cfgLoad');
const cfgReset = document.getElementById('cfgReset');

// Folder fields (in header)
const outputBtn = document.getElementById('outputBtn');
const outputPath = document.getElementById('outputPath');
const tempBtn = document.getElementById('tempBtn');
const tempPath = document.getElementById('tempPath');

// Detected / Restored thumbnails + zoom
const targetImg = document.getElementById('targetImg');
const swappedImg = document.getElementById('swappedImg');
const zoomBtn = document.getElementById('zoomBtn');

// Center
const centerArea = document.getElementById('centerArea');
const videoDrop = document.getElementById('videoDrop');
const videoContainer = document.getElementById('videoContainer');
const player = document.getElementById('player');
const zoomOverlay = document.getElementById('zoomOverlay');
const zoomOriginal = document.getElementById('zoomOriginal');
const zoomImage = document.getElementById('zoomImage');
const zoomClose = document.getElementById('zoomClose');

// Footer — transport controls (matching Tilester layout)
const currentTime = document.getElementById('currentTime');
const copyTimeBtn = document.getElementById('copyTimeBtn');
const frameNum = document.getElementById('frameNum');
const segMarkBtn = document.getElementById('segMarkBtn');
const fpsDisplay = document.getElementById('fpsDisplay');
const pinBtn = document.getElementById('pinBtn');
const skipBackward = document.getElementById('skipBackward');
const skipForward = document.getElementById('skipForward');
const gotoFrame = document.getElementById('gotoFrame');
const gotoTime = document.getElementById('gotoTime');
const gotoFrameBtn = document.getElementById('gotoFrameBtn');
const gotoTimeBtn = document.getElementById('gotoTimeBtn');
const newBtn = document.getElementById('newBtn');
const detectBtn = document.getElementById('detectBtn');
const swapBtn = document.getElementById('swapBtn');
const previewBtn = document.getElementById('previewBtn');
const swapSaveBtn = document.getElementById('swapSaveBtn');
const fsBtn = document.getElementById('fsBtn');

// Progress modal
const progressModal = document.getElementById('progressModal');
const progressTitle = document.getElementById('progressTitle');
const progressBar = document.getElementById('progressBar');
const progressParams = document.getElementById('progressParams');
const progressPercent = document.getElementById('progressPercent');
const progressFps = document.getElementById('progressFps');
const progressEta = document.getElementById('progressEta');
const progressCancel = document.getElementById('progressCancel');

// Confirm modal
const confirmModal = document.getElementById('confirmModal');
const confirmTitle = document.getElementById('confirmTitle');
const confirmMessage = document.getElementById('confirmMessage');
const confirmYes = document.getElementById('confirmYes');
const confirmNo = document.getElementById('confirmNo');

// ── Player State (used by player.js) ─────────────────────
let fps = null;

// ── Application State ────────────────────────────────────
const state = {
  videoPath: null,
  videoInfo: null,
  currentFrame: 0,

  facesDir: '',
  sourceFaces: [],

  detectedFaces: [],
  currentFaceIdx: 0,

  assignments: {},
  hasSwappedImage: false,

  outputDir: '',

  // Segment (set by segMarkBtn in player.js)
  segmentStartTime: 0,
  segmentEndTime: 0,
  segmentStartFrame: 0,
  segmentEndFrame: 0,

  previewReady: false,
  previewMode: false,
  zoomOpen: false,
};