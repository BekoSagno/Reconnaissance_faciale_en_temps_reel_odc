const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models")
]).then(startVideo);

function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => console.error("Erreur webcam:", err));
}

video.addEventListener("play", async () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  // Charger les visages connus
  const labeledDescriptors = await loadKnownFaces();

  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    const results = resizedDetections.map(d =>
      faceMatcher.findBestMatch(d.descriptor)
    );

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
      drawBox.draw(canvas);
    });

    statusEl.innerText = results.length > 0
      ? "Visages détectés"
      : "Aucun visage détecté";
  }, 100);
});

// Charger les images des visages connus
async function loadKnownFaces() {
  const faces = {
    "Bekosagno": [
      "beko1.jpg", "beko2.jpg", "beko3.jpg", "beko4.jpg", "beko5.jpg",
      "beko6.jpg", "beko7.jpg", "beko8.jpg", "beko9.jpg", "beko10.jpg"
    ],
    "Moustapha": [
      "moustapha1.jpg", "moustapha2.jpg", "moustapha3.jpg", "moustapha4.jpg", "moustapha5.jpg"
    ]
  };

  return Promise.all(
    Object.keys(faces).map(async label => {
      const descriptors = [];
      for (const file of faces[label]) {
        try {
          const imgUrl = `/known_faces/${label}/${file}`;
          const img = await faceapi.fetchImage(imgUrl);
          const detection = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (detection) descriptors.push(detection.descriptor);
        } catch (err) {
          console.warn(`⚠️ Erreur avec l'image ${label}/${file}:`, err);
        }
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptors);
    })
  );
}
