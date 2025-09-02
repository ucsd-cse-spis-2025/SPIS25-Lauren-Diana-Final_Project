let startY = 0;

const intro = document.getElementById('intro');
const main = document.getElementById('main');

// Touch events for mobile
intro.addEventListener('touchstart', (e) => {
  startY = e.touches[0].clientY;
});

intro.addEventListener('touchend', (e) => {
  let endY = e.changedTouches[0].clientY;
  if (startY - endY > 50) {
    slideUp();
  }
});

// Mouse events for desktop
intro.addEventListener('mousedown', (e) => {
  startY = e.clientY;
});

intro.addEventListener('mouseup', (e) => {
  let endY = e.clientY;
  if (startY - endY > 50) {
    slideUp();
  }
});

// Function to slide up
function slideUp() {
  intro.style.transition = 'transform 0.5s ease';
  intro.style.transform = 'translateY(-100vh)';

  setTimeout(() => {
    intro.style.display = 'none';
    main.style.display = 'block';
  }, 500);
}
