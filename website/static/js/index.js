document.addEventListener('DOMContentLoaded', () => {
  // Use IntersectionObserver to lazily load videos when they come into view
  const lazyVideoObserver = new IntersectionObserver(function(entries, observer) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        const video = entry.target;
        video.src = video.dataset.src;
        video.load();
        observer.unobserve(entry.target);
      }
    });
  });
  document.querySelectorAll('video[data-src]').forEach(video => lazyVideoObserver.observe(video));

  // Tiles with data-video swap their still (or placeholder) for the clip once it exists
  document.querySelectorAll('[data-video]').forEach(slot => {
    const video = document.createElement('video');
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.setAttribute('playsinline', '');
    video.addEventListener('loadeddata', () => {
      slot.innerHTML = '';
      slot.classList.remove('ph');
      slot.appendChild(video);
    });
    video.src = slot.dataset.video;
  });
});
