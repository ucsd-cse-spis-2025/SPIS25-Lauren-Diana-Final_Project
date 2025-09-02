let startY = 0;

document.getElementById('intro').addEventListener('touchstart', (e) => {
  startY = e.touches[0].clientY;
});

document.getElementById('intro').addEventListener('touchend', (e) => {
  let endY = e.changedTouches[0].clientY;
  if (startY - endY > 50) { // swipe up
    document.getElementById('intro').style.transform = 'translateY(-100vh)';
    document.getElementById('main').style.transform = 'translateY(-100vh)';
  }
});
