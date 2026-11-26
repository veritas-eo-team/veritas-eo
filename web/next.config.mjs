/** @type {import('next').NextConfig} */
const API = process.env.VERITAS_API || "http://127.0.0.1:8077";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
