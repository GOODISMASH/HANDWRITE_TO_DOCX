const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewWrap = document.getElementById("previewWrap");
const formError = document.getElementById("formError");

if (imageInput && imagePreview && previewWrap) {
    imageInput.addEventListener("change", () => {
        const file = imageInput.files && imageInput.files[0];

        if (!file) {
            previewWrap.hidden = true;
            imagePreview.removeAttribute("src");
            return;
        }

        formError.textContent = "";
        imagePreview.src = URL.createObjectURL(file);
        previewWrap.hidden = false;
    });
}

if (uploadForm && imageInput && formError) {
    uploadForm.addEventListener("submit", (event) => {
        if (!imageInput.files || imageInput.files.length === 0) {
            event.preventDefault();
            formError.textContent = "Выберите файл перед отправкой.";
        }
    });
}

const segmentationCanvas = document.getElementById("segmentationCanvas");
const manualCutsInput = document.getElementById("manualCutsInput");
const manualCutsData = document.getElementById("manualCutsData");
const undoCutButton = document.getElementById("undoCutButton");
const resetCutsButton = document.getElementById("resetCutsButton");
const clearCutsButton = document.getElementById("clearCutsButton");

if (segmentationCanvas && manualCutsInput && manualCutsData) {
    const context = segmentationCanvas.getContext("2d");
    const sourceImage = new Image();
    const initialCuts = JSON.parse(manualCutsData.textContent || "[]");
    let cuts = [...initialCuts].sort((first, second) => first - second);

    function syncCuts() {
        manualCutsInput.value = JSON.stringify(cuts);
    }

    function drawSegmentation() {
        if (!sourceImage.complete || !sourceImage.naturalWidth) {
            return;
        }

        segmentationCanvas.width = sourceImage.naturalWidth;
        segmentationCanvas.height = sourceImage.naturalHeight;
        context.drawImage(sourceImage, 0, 0);
        context.strokeStyle = "#ef4056";
        context.fillStyle = "#ef4056";
        context.lineWidth = Math.max(2, sourceImage.naturalHeight / 180);
        context.font = `${Math.max(14, sourceImage.naturalHeight / 26)}px Segoe UI, Arial`;

        cuts.forEach((position, index) => {
            const y = position * segmentationCanvas.height;
            context.beginPath();
            context.moveTo(0, y);
            context.lineTo(segmentationCanvas.width, y);
            context.stroke();
            context.fillText(`${index + 1}`, 8, Math.max(18, y - 7));
        });

        syncCuts();
    }

    syncCuts();
    sourceImage.addEventListener("load", drawSegmentation);
    sourceImage.src = segmentationCanvas.dataset.imageUrl;

    segmentationCanvas.addEventListener("click", (event) => {
        const bounds = segmentationCanvas.getBoundingClientRect();
        const position = (event.clientY - bounds.top) / bounds.height;
        const hitRange = 10 / bounds.height;
        const closestIndex = cuts.findIndex((cut) => Math.abs(cut - position) <= hitRange);

        if (closestIndex >= 0) {
            cuts.splice(closestIndex, 1);
        } else if (position > 0.01 && position < 0.99) {
            cuts.push(position);
            cuts.sort((first, second) => first - second);
        }

        drawSegmentation();
    });

    if (undoCutButton) {
        undoCutButton.addEventListener("click", () => {
            cuts.pop();
            drawSegmentation();
        });
    }

    if (resetCutsButton) {
        resetCutsButton.addEventListener("click", () => {
            cuts = [...initialCuts];
            drawSegmentation();
        });
    }

    if (clearCutsButton) {
        clearCutsButton.addEventListener("click", () => {
            cuts = [];
            drawSegmentation();
        });
    }
}
