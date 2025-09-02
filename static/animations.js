// Brain Tumor System - Animation JavaScript

document.addEventListener("DOMContentLoaded", function () {
  // Initialize all animations
  initScrollAnimations();
});

// Scroll-triggered animations
function initScrollAnimations() {
  const animateElements = document.querySelectorAll(".animate-on-scroll");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          // Optional: unobserve after animation
          // observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.1, // Trigger when 10% of the element is visible
      rootMargin: "0px 0px -50px 0px", // Start animation a bit later
    }
  );

  animateElements.forEach((el) => {
    observer.observe(el);
  });
}
