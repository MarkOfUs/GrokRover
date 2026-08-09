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

  // Placeholder tiles with data-video swap themselves for the clip once it exists
  document.querySelectorAll('.ph[data-video]').forEach(ph => {
    const video = document.createElement('video');
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.setAttribute('playsinline', '');
    video.addEventListener('loadeddata', () => ph.replaceWith(video));
    video.src = ph.dataset.video;
  });
});
