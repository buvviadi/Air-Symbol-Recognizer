import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

const traceCanvas = document.querySelector('#trace-canvas');
const traceContext = traceCanvas.getContext('2d');
const drawingStage = document.querySelector('#drawing-stage');
const emptyState = document.querySelector('#stage-empty');
const objectCanvas = document.querySelector('#object-canvas');
const prediction = document.querySelector('#prediction');
const confidence = document.querySelector('#confidence');
const description = document.querySelector('#description');
const modeLabel = document.querySelector('#mode-label');
const objectName = document.querySelector('#object-name');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
const renderer = new THREE.WebGLRenderer({ canvas: objectCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
camera.position.set(0, 0, 8);
const objectRoot = new THREE.Group();
scene.add(objectRoot);
scene.add(new THREE.AmbientLight(0x8ffff0, 1.4));
const keyLight = new THREE.DirectionalLight(0x9ffff4, 3.2);
keyLight.position.set(3, 4, 6);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0xff3b9f, 8, 12);
rimLight.position.set(-3, 1, 3);
scene.add(rimLight);
let targetRotation = 0;
let targetPitch = -.25;
let targetYaw = .4;
let dragStart = null;

function material(color, metalness = .55, roughness = .24) {
  return new THREE.MeshStandardMaterial({ color, metalness, roughness });
}

function cylinder(radius, height, color, radialSegments = 32) {
  return new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, radialSegments), material(color));
}

function buildObject(label) {
  objectRoot.clear();
  const cyan = 0x42ded0;
  const dark = 0x102d30;
  if (label === 'gear') {
    const body = new THREE.Mesh(new THREE.CylinderGeometry(1.35, 1.35, .48, 32), material(cyan));
    body.rotation.x = Math.PI / 2;
    objectRoot.add(body);
    for (let i = 0; i < 12; i += 1) {
      const tooth = new THREE.Mesh(new THREE.BoxGeometry(.34, .34, .58), material(cyan));
      const angle = i * Math.PI / 6;
      tooth.position.set(Math.cos(angle) * 1.42, Math.sin(angle) * 1.42, 0);
      tooth.rotation.z = angle;
      objectRoot.add(tooth);
    }
    const hole = new THREE.Mesh(new THREE.CylinderGeometry(.4, .4, .62, 32), material(dark));
    hole.rotation.x = Math.PI / 2;
    objectRoot.add(hole);
  } else if (label === 'pendulum') {
    const rod = cylinder(.075, 3.2, cyan);
    rod.position.y = -.2;
    objectRoot.add(rod);
    const weight = new THREE.Mesh(new THREE.SphereGeometry(.55, 32, 20), material(0x56e7d4));
    weight.position.y = -1.9;
    objectRoot.add(weight);
    const mount = new THREE.Mesh(new THREE.SphereGeometry(.16, 20, 14), material(0xe6fffb));
    mount.position.y = 1.4;
    objectRoot.add(mount);
  } else if (label === 'angle') {
    const left = cylinder(.1, 2.6, 0xd95a9e);
    left.rotation.z = Math.PI / 4;
    left.position.set(-.75, .58, 0);
    objectRoot.add(left);
    const right = cylinder(.1, 2.6, cyan);
    right.rotation.z = -Math.PI / 4;
    right.position.set(.75, .58, 0);
    objectRoot.add(right);
    objectRoot.add(new THREE.Mesh(new THREE.SphereGeometry(.16, 20, 14), material(0xe6fffb)));
  } else {
    const body = cylinder(.42, 2.3, 0xd8c2a3, 40);
    body.rotation.z = Math.PI / 2;
    objectRoot.add(body);
    [0x8b4d38, 0x17191c, 0xd95a9e].forEach((color, index) => {
      const band = cylinder(.45, .18, color, 40);
      band.rotation.z = Math.PI / 2;
      band.position.x = -.65 + index * .65;
      objectRoot.add(band);
    });
    [-1.5, 1.5].forEach((x) => {
      const lead = cylinder(.07, .65, 0xb6d5d1, 20);
      lead.rotation.z = Math.PI / 2;
      lead.position.x = x;
      objectRoot.add(lead);
    });
  }
  objectRoot.rotation.set(targetPitch, targetYaw, targetRotation);
}

function resizeObjectStage() {
  const rect = objectCanvas.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}

function renderObject() {
  objectRoot.rotation.z += (targetRotation - objectRoot.rotation.z) * .08;
  objectRoot.rotation.x += (targetPitch - objectRoot.rotation.x) * .08;
  objectRoot.rotation.y += (targetYaw - objectRoot.rotation.y) * .08;
  renderer.render(scene, camera);
  requestAnimationFrame(renderObject);
}

window.addEventListener('resize', resizeObjectStage);
objectCanvas.addEventListener('pointerdown', (event) => { dragStart = event.clientX; objectCanvas.setPointerCapture(event.pointerId); });
objectCanvas.addEventListener('pointermove', (event) => { if (dragStart !== null) { targetRotation += (event.clientX - dragStart) * .012; dragStart = event.clientX; } });
objectCanvas.addEventListener('pointerup', () => { dragStart = null; });
resizeObjectStage();
buildObject('resistor');
renderObject();

function showTrace(encoded) {
  if (!encoded) return;
  const image = new Image();
  image.onload = () => {
    traceContext.clearRect(0, 0, traceCanvas.width, traceCanvas.height);
    traceContext.drawImage(image, 0, 0, traceCanvas.width, traceCanvas.height);
    emptyState.style.display = 'none';
  };
  image.src = `data:image/jpeg;base64,${encoded}`;
}

function applyState(state) {
  const hasResult = state.mode === 'rotate' && Boolean(state.prediction);
  drawingStage.classList.toggle('is-result', hasResult);
  if (hasResult) resizeObjectStage();
  prediction.textContent = state.prediction ? state.prediction.charAt(0).toUpperCase() + state.prediction.slice(1) : 'Awaiting gesture';
  confidence.textContent = state.prediction ? `${state.confidence}% confidence` : 'Draw a symbol, then classify';
  description.textContent = state.prediction ? state.description : 'Draw a symbol to see its description.';
  modeLabel.textContent = state.mode === 'rotate' ? 'ROTATE MODE' : 'DRAW MODE';
  if (state.mode === 'rotate' && state.rotation) {
    targetPitch = -.25 + state.rotation.x;
    targetYaw = .4 + state.rotation.y;
  }
  if (state.prediction) {
    buildObject(state.prediction);
    objectName.textContent = `3D OBJECT / ${state.prediction.toUpperCase()}`;
    emptyState.classList.add('has-object');
  } else {
    objectName.textContent = '3D OBJECT / READY';
    emptyState.classList.remove('has-object');
  }
  showTrace(state.trace);
}

async function updateState() {
  try {
    const response = await fetch('/api/state');
    applyState(await response.json());
  } catch (error) {
    confidence.textContent = 'Vision service unavailable';
  }
}

document.querySelector('#clear-button').addEventListener('click', async () => {
  applyState(await (await fetch('/api/clear', { method: 'POST' })).json());
  traceContext.clearRect(0, 0, traceCanvas.width, traceCanvas.height);
  emptyState.style.display = 'grid';
});

document.querySelector('#classify-button').addEventListener('click', async () => {
  const button = document.querySelector('#classify-button');
  button.disabled = true;
  button.textContent = 'Analysing...';
  applyState(await (await fetch('/api/classify', { method: 'POST' })).json());
  button.disabled = false;
  button.innerHTML = 'Classify gesture <span>↗</span>';
});

document.querySelector('#export-button').addEventListener('click', () => {
  traceCanvas.toBlob((blob) => {
    const link = document.createElement('a');
    link.download = 'gesture-session.png';
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
  });
});

updateState();
setInterval(updateState, 100);
