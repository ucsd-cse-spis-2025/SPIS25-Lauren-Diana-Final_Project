let startY = 0;
let isDragging = false;

const intro = document.getElementById('intro');
const main = document.getElementById('main');

// Function to slide up
function slideUp() {
    intro.style.transition = 'transform 0.5s ease';
    intro.style.transform = 'translateY(-100vh)';

    setTimeout(() => {
        intro.style.display = 'none';
        main.style.display = 'block';
    }, 500);
}

// ================= Touchscreen swipe =================
intro.addEventListener('touchstart', (e) => {
    startY = e.touches[0].clientY;
});

intro.addEventListener('touchend', (e) => {
    let endY = e.changedTouches[0].clientY;
    if (startY - endY > 50) { // swipe up
        slideUp();
    }
});

// ================= Mouse drag swipe =================
intro.addEventListener('mousedown', (e) => {
    isDragging = true;
    startY = e.clientY;
});

intro.addEventListener('mouseup', (e) => {
    if (!isDragging) return;
    isDragging = false;
    let endY = e.clientY;
    if (startY - endY > 50) { // drag up
        slideUp();
    }
});

// ================= Trackpad scroll swipe =================
intro.addEventListener('wheel', (e) => {
    if (e.deltaY > 50) { // scroll down
        // ignore
        return;
    }
    if (e.deltaY < -50) { // scroll up
        slideUp();
    }
});

// ================= Optional click to skip =================
const skipBtn = document.getElementById('skip-intro');
if (skipBtn) {
    skipBtn.addEventListener('click', slideUp);
}
