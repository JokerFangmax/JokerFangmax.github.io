---
title: "课程大作业示例：光线追踪渲染器"
summary: "基于 Whitted 风格的光线追踪器实现，支持反射、折射、抗锯齿和多种材质。"
authors: []
tags: ["Computer Graphics", "C++", "Ray Tracing"]
categories: []
date: 2025-12-15

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Demo
#   url: "https://your-demo-link.com"
#   icon_pack: fab
#   icon: youtube
# - name: Code
#   url: "https://github.com/yourname/your-repo"
#   icon_pack: fab
#   icon: github

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

## 项目简介

这是我的计算机图形学课程大作业。实现了一个基于 Whitted 风格的光线追踪器，主要特性包括：

- **光线追踪核心**：支持递归光线追踪，可处理反射和折射
- **材质系统**：实现了漫反射、镜面反射、玻璃等多种材质
- **抗锯齿**：使用超采样技术减少锯齿效应
- **加速结构**：使用 BVH (Bounding Volume Hierarchy) 加速光线与场景的求交

## 运行方式

```bash
git clone https://github.com/yourname/raytracer.git
cd raytracer
mkdir build && cd build
cmake ..
make -j
./raytracer ../scenes/cornell_box.yaml
```

## 演示视频

你可以在以下链接查看演示视频：

- [Bilibili 链接](https://www.bilibili.com/video/xxx)
- [YouTube 链接](https://youtube.com/watch?v=xxx)

## 技术细节

### 光线-物体求交

对于球体，光线求交方程为：

$$
\|\mathbf{o} + t\mathbf{d} - \mathbf{c}\|^2 = r^2
$$

其中 $\mathbf{o}$ 是光线原点，$\mathbf{d}$ 是光线方向，$\mathbf{c}$ 是球心，$r$ 是半径。

### 反射方向计算

反射方向根据入射方向和法线计算：

$$
\mathbf{r} = \mathbf{i} - 2(\mathbf{i} \cdot \mathbf{n})\mathbf{n}
$$

## 截图展示

你可以在页面同级目录添加 `featured.png` 或 `screenshot1.png` 等图片来展示效果。
