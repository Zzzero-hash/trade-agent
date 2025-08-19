import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef9ff",
          100: "#d9f0ff",
          200: "#b3e1ff",
          300: "#7ccaff",
          400: "#36abff",
          500: "#068dff",
          600: "#006fde",
          700: "#0057b1",
          800: "#004a92",
          900: "#003365",
        },
      },
    },
  },
  plugins: [],
};

export default config;
