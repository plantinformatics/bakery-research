import { withAui } from "@assistant-ui/next";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  agentRules: false,
};

export default withAui(nextConfig);
