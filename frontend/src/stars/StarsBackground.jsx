import React, { useEffect, useRef, useCallback } from 'react';
import './StarsBackground.css';

const StarsBackground = ({
  starCount = 150,
  maxDistance = 250,
  className = '',
  children,
}) => {
  const canvasRef = useRef(null);
  const starsRef = useRef([]);
  const animationFrameRef = useRef();
  const mouseRef = useRef({ x: -1000, y: -1000 });

  const initStars = useCallback((width, height) => {
    const stars = [];

    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        originalX: 0,
        originalY: 0,
        speed: Math.random() * 0.02 + 0.01,
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        twinkleOffset: Math.random() * Math.PI * 2,
      });
    }

    stars.forEach((star) => {
      star.originalX = star.x;
      star.originalY = star.y;
    });

    starsRef.current = stars;
  }, [starCount]);

  const drawStars = useCallback(
    (ctx, width, height, time) => {
      ctx.clearRect(0, 0, width, height);

      const mouseX = mouseRef.current.x;
      const mouseY = mouseRef.current.y;

      starsRef.current.forEach((star) => {
        const dx = mouseX - star.originalX;
        const dy = mouseY - star.originalY;

        const distance = Math.sqrt(dx * dx + dy * dy);

        const force = Math.max(0, 1 - distance / maxDistance);

        if (distance < maxDistance) {
          const attractionStrength = force * 0.8;
          const targetX = star.originalX + dx * attractionStrength;
          const targetY = star.originalY + dy * attractionStrength;

          star.x += (targetX - star.x) * 0.15;
          star.y += (targetY - star.y) * 0.15;
        } else {
          star.x += (star.originalX - star.x) * 0.08;
          star.y += (star.originalY - star.y) * 0.08;
        }

        const twinkle = Math.sin(time * star.twinkleSpeed + star.twinkleOffset) * 0.3 + 0.7;
        const opacity = star.opacity * twinkle * (1 + force * 0.5);

        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size * (1 + force * 0.5), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
        ctx.fill();
      });
    },
    [maxDistance]
  );

  const handleMouseMove = useCallback((event) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    mouseRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }, []);

  const handleMouseLeave = useCallback(() => {
    mouseRef.current = { x: -1000, y: -1000 };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initStars(canvas.width, canvas.height);
    };

    resizeCanvas();

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);

    const animate = (time) => {
      drawStars(ctx, canvas.width, canvas.height, time);
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [initStars, drawStars, handleMouseMove, handleMouseLeave]);

  return (
    <div className={`stars-background-container ${className}`}>
      <canvas ref={canvasRef} className="stars-canvas" />
      {children && <div className="stars-content">{children}</div>}
    </div>
  );
};

export default StarsBackground;