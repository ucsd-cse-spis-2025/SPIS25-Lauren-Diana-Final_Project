let startY = 0;

const intro = document.getElementById('intro');
const main = document.getElementById('main');

intro.addEventListener('touchstart', (e) => {
  startY = e.touches[0].clientY;
});

intro.addEventListener('touchend', (e) => {
  let endY = e.changedTouches[0].clientY;
  if (startY - endY > 50) { // swipe up
    intro.style.transition = 'transform 0.5s ease';
    intro.style.transform = 'translateY(-100vh)'; // move intro up

    setTimeout(() => {
      intro.style.display = 'none';  // hide intro
      main.style.display = 'block';  // show main page
    }, 500); // matches transition duration
  }
});
