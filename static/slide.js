const intro = document.getElementById('intro');
const main = document.getElementById('main');

// Function to slide up
function slideUp() {
    intro.style.transition = 'transform 0.5s ease';
    intro.style.transform = 'translateY(-100vh)';

    setTimeout(() => {
        intro.style.display = 'none';
        main.style.display = 'block';
    }, 500); // matches transition duration
}

// Click anywhere on intro to slide up
intro.addEventListener('click', slideUp);


