import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import mediapipe as mp
import numpy as np
import torch
from model import SymbolCNN

IMAGE_SIZE = 64

with open("labels.txt", "r") as f:
    symbols_list = [line.strip() for line in f.readlines()]

model = SymbolCNN(num_classes=len(symbols_list))
model.load_state_dict(torch.load("Symbol_model.pth", map_location="cpu"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

success, frame = cap.read()
cam_h, cam_w, _ = frame.shape
draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
prev_hand_x = None
mode = "draw"
current_label = None
smooth_rot = 30.0

ACCENT = (0, 220, 255)
ACCENT_DIM = (0, 120, 160)
SIGNAL = (255, 30, 145)
TEXT_MAIN = (225, 240, 255)
TEXT_DIM = (150, 175, 200)
INK = (5, 7, 12)

components_data = {
    "resistor": "An electronic component that\nlimits electrical current.\nValue: 220 Ohm +/-5%",
    "gear": "A rotating part with teeth\nthat transmits force and speed.\nMaterial: Steel",
    "pendulum": "A weight on a rod that swings\nfreely under gravity.\nDemonstrates harmonic motion.",
    "angle": "The space between two lines\nthat meet at a point.\nMeasured in degrees.",
}

def draw_cylinder(radius, length, color):
    glColor3f(*color)
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluCylinder(quad, radius, radius, length, 28, 1)
    glPushMatrix()
    glRotatef(180, 1, 0, 0)
    gluDisk(quad, 0, radius, 28, 1)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0, 0, length)
    gluDisk(quad, 0, radius, 28, 1)
    glPopMatrix()

def draw_resistor():
    glPushMatrix()
    glTranslatef(0, 0, -1.5)
    draw_cylinder(0.04, 1.0, (0.8, 0.8, 0.8))
    glPushMatrix()
    glTranslatef(0, 0, 1.0)
    draw_cylinder(0.35, 1.0, (0.8, 0.6, 0.4))
    glPushMatrix(); glTranslatef(0, 0, 0.2); draw_cylinder(0.36, 0.15, (0.6, 0.1, 0.1)); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0, 0.5); draw_cylinder(0.36, 0.15, (0.1, 0.1, 0.1)); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0, 0.8); draw_cylinder(0.36, 0.15, (0.8, 0.4, 0.1)); glPopMatrix()
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0, 0, 2.0)
    draw_cylinder(0.04, 1.0, (0.8, 0.8, 0.8))
    glPopMatrix()
    glPopMatrix()

def draw_pendulum():
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.5)
    quad = gluNewQuadric()
    gluSphere(quad, 0.1, 20, 20)
    glRotatef(90, 1, 0, 0)
    draw_cylinder(0.05, 2.5, (0.8, 0.8, 0.2))
    glPushMatrix()
    glTranslatef(0, 0, 2.5)
    glColor3f(0.2, 0.6, 1.0)
    gluSphere(quad, 0.4, 32, 32)
    glPopMatrix()
    glPopMatrix()

def draw_gear():
    glPushMatrix()
    glTranslatef(0, 0, -0.2)
    draw_cylinder(1.0, 0.4, (0.5, 0.5, 0.58))
    teeth_count = 12
    for i in range(teeth_count):
        glPushMatrix()
        angle = (360 / teeth_count) * i
        glRotatef(angle, 0, 0, 1)
        glTranslatef(0.9, 0, 0)
        draw_cylinder(0.2, 0.4, (0.5, 0.5, 0.58))
        glPopMatrix()
    glTranslatef(0, 0, -0.01)
    draw_cylinder(0.3, 0.42, (0.08, 0.08, 0.1))
    glPopMatrix()

def draw_angle():
    glPushMatrix()
    glPushMatrix()
    glRotatef(-20, 0, 1, 0)
    draw_cylinder(0.05, 1.5, (0.8, 0.3, 0.6))
    glPopMatrix()
    glPushMatrix()
    glRotatef(20, 0, 1, 0)
    draw_cylinder(0.05, 1.5, (0.3, 0.6, 0.8))
    glPopMatrix()
    glPopMatrix()

shape_functions = {
    "resistor": draw_resistor,
    "gear": draw_gear,
    "pendulum": draw_pendulum,
    "angle": draw_angle,
}

shape_scales = {
    "resistor": 0.82,
    "gear": 0.92,
    "pendulum": 0.88,
    "angle": 1.15,
}

def draw_corner_brackets(surface, rect, size=14, thickness=2, accent=ACCENT):
    x, y, w, h = rect
    c = accent
    corners = [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]
    for cx, cy, dx, dy in corners:
        pygame.draw.line(surface, c, (cx, cy), (cx + dx * size, cy), thickness)
        pygame.draw.line(surface, c, (cx, cy), (cx, cy + dy * size), thickness)

def draw_glass_panel(surface, rect, title=None, fill_alpha=150, accent=ACCENT):
    x, y, w, h = rect
    glow = pygame.Surface((w + 16, h + 16), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*accent, 18), (0, 0, w + 16, h + 16), border_radius=12)
    surface.blit(glow, (x - 8, y - 8))

    glass = pygame.Surface((w, h), pygame.SRCALPHA)
    glass.fill((*INK, fill_alpha))
    surface.blit(glass, (x, y))

    pygame.draw.rect(surface, (*accent, 110), (x, y, w, h), 1, border_radius=8)
    draw_corner_brackets(surface, rect, accent=accent)

    if title:
        font = pygame.font.SysFont("bahnschrift", 14, bold=True)
        text = font.render(title.upper(), True, accent)
        surface.blit(text, (x + 18, y + 12))
        pygame.draw.line(surface, (*accent, 75), (x + 15, y + 36), (x + w - 15, y + 36), 1)

def frame_to_surface(frame_bgr, target_size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame_rgb)
    return pygame.transform.scale(surf, target_size)

def draw_sci_fi_backdrop(surface, width, height):
    # Keep the full-screen layer transparent so the 3D object remains visible.
    grid = pygame.Surface((width, height), pygame.SRCALPHA)
    for x in range(18, width, 46):
        pygame.draw.line(grid, (85, 110, 135, 20), (x, 0), (x, height), 1)
    for y in range(18, height, 46):
        pygame.draw.line(grid, (85, 110, 135, 20), (0, y), (width, y), 1)
    for x in range(18, width, 46):
        for y in range(18, height, 46):
            pygame.draw.circle(grid, (180, 195, 215, 70), (x, y), 1)
    surface.blit(grid, (0, 0))

    bands = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.polygon(bands, (45, 115, 150, 16), [(0, 470), (0, 590), (720, 0), (590, 0)])
    pygame.draw.polygon(bands, (220, 25, 120, 10), [(760, 0), (960, 0), (1280, 310), (1280, 230)])
    pygame.draw.line(bands, (105, 190, 220, 58), (0, 470), (720, 0), 1)
    pygame.draw.line(bands, (255, 30, 145, 52), (760, 0), (1280, 310), 1)
    surface.blit(bands, (0, 0))

def render_hud(width, height, live_frame):
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    draw_sci_fi_backdrop(surface, width, height)

    font_display = pygame.font.SysFont("bahnschrift", 28, bold=True)
    font_title = pygame.font.SysFont("bahnschrift", 20, bold=True)
    font_body = pygame.font.SysFont("segoeui", 16)
    font_small = pygame.font.SysFont("consolas", 12)

    # Top navigation gives the tool the calm, product-like framing of the reference.
    nav_rect = (150, 16, 980, 60)
    draw_glass_panel(surface, nav_rect, fill_alpha=92)
    pygame.draw.circle(surface, SIGNAL, (252, 43), 4)
    brand = font_title.render("AIR / DRAW", True, TEXT_MAIN)
    surface.blit(brand, (266, 28))
    brand_sub = font_small.render("VISUAL INSTRUMENTS", True, TEXT_DIM)
    surface.blit(brand_sub, (380, 34))
    nav_items = [(530, "WORKSPACE"), (650, "METHOD"), (755, "COMPONENTS"), (900, "SIGNALS"), (1005, "ABOUT")]
    for x, item in nav_items:
        nav_text = font_small.render(item, True, TEXT_DIM)
        surface.blit(nav_text, (x, 36))
    pygame.draw.circle(surface, (40, 220, 160), (1090, 42), 4)
    status = font_small.render("SYSTEM ONLINE", True, TEXT_MAIN)
    surface.blit(status, (1102, 35))

    # Main stage: its low alpha lets the OpenGL object remain the visual focus.
    viewport_rect = (150, 108, 650, 420)
    draw_glass_panel(surface, viewport_rect, "OBJECT / LIVE VIEW", fill_alpha=42)
    pygame.draw.line(surface, (*ACCENT, 80), (viewport_rect[0] + 20, viewport_rect[1] + 45),
                     (viewport_rect[0] + 178, viewport_rect[1] + 45), 1)
    stage_label = current_label.upper() if current_label else "AWAITING INPUT"
    object_text = font_display.render(stage_label, True, TEXT_MAIN if current_label else TEXT_DIM)
    surface.blit(object_text, (viewport_rect[0] + 22, viewport_rect[1] + 62))
    stage_meta = font_small.render("ROTATION / HAND X-AXIS     DEPTH / 0.1 - 50.0", True, TEXT_DIM)
    surface.blit(stage_meta, (viewport_rect[0] + 23, viewport_rect[1] + 99))
    for y in range(viewport_rect[1] + 144, viewport_rect[1] + viewport_rect[3] - 22, 52):
        pygame.draw.line(surface, (130, 170, 190, 30), (viewport_rect[0] + 22, y),
                         (viewport_rect[0] + viewport_rect[2] - 22, y), 1)
    pygame.draw.line(surface, (130, 170, 190, 45), (viewport_rect[0] + viewport_rect[2] // 2, viewport_rect[1] + 140),
                     (viewport_rect[0] + viewport_rect[2] // 2, viewport_rect[1] + viewport_rect[3] - 22), 1)
    pygame.draw.arc(surface, (*ACCENT, 65), (viewport_rect[0] + 178, viewport_rect[1] + 145, 294, 294), 0.25, 2.9, 1)
    pygame.draw.circle(surface, (*SIGNAL, 140), (viewport_rect[0] + 325, viewport_rect[1] + 292), 3)

    desc_rect = (840, 108, 330, 210)
    draw_glass_panel(surface, desc_rect, "IDENTIFICATION", fill_alpha=168, accent=SIGNAL)
    if current_label:
        desc = components_data[current_label]
        for i, line in enumerate(desc.split("\n")):
            text = font_body.render(line, True, TEXT_MAIN)
            surface.blit(text, (desc_rect[0] + 20, desc_rect[1] + 60 + i * 25))
    else:
        text = font_body.render("Draw a symbol, then press P", True, TEXT_DIM)
        surface.blit(text, (desc_rect[0] + 20, desc_rect[1] + 60))
    confidence = font_small.render("CLASSIFIER / SYMBOL CNN", True, TEXT_DIM)
    surface.blit(confidence, (desc_rect[0] + 20, desc_rect[1] + desc_rect[3] - 28))

    cam_rect = (840, 346, 330, 190)
    draw_glass_panel(surface, cam_rect, "INPUT / CAMERA", fill_alpha=174)
    cam_surf = frame_to_surface(live_frame, (290, 122))
    surface.blit(cam_surf, (cam_rect[0] + 20, cam_rect[1] + 50))
    pygame.draw.rect(surface, (*ACCENT, 80), (cam_rect[0] + 20, cam_rect[1] + 50, 290, 122), 1)

    canvas_rect = (150, 556, 310, 132)
    draw_glass_panel(surface, canvas_rect, "TRACE / INPUT", fill_alpha=150)
    draw_surf = frame_to_surface(draw_canvas, (270, 78))
    surface.blit(draw_surf, (canvas_rect[0] + 20, canvas_rect[1] + 42))

    rail_rect = (480, 556, 690, 132)
    draw_glass_panel(surface, rail_rect, "CONTROL SURFACE", fill_alpha=128)
    mode_text = "DRAW MODE" if mode == "draw" else "ROTATE MODE"
    mode_surf = font_title.render(mode_text, True, ACCENT)
    surface.blit(mode_surf, (rail_rect[0] + 22, rail_rect[1] + 48))
    command = "P  CLASSIFY" if mode == "draw" else "MOVE HAND  ROTATE    C  RESET"
    command_surf = font_small.render(command, True, TEXT_MAIN)
    surface.blit(command_surf, (rail_rect[0] + 250, rail_rect[1] + 55))
    pygame.draw.line(surface, (*SIGNAL, 180), (rail_rect[0] + 210, rail_rect[1] + 44),
                     (rail_rect[0] + 210, rail_rect[1] + 88), 1)

    footer = font_small.render("AIR-DRAW / 01     REAL-TIME SYMBOL RECOGNITION", True, TEXT_DIM)
    surface.blit(footer, (38, height - 25))

    # Flip once during upload; the fullscreen quad uses bottom-left OpenGL UVs.
    texture_data = pygame.image.tostring(surface, "RGBA", 1)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    return tex_id

def draw_hud_fullscreen(tex_id, w, h):
    # OpenGL projection setup: configure perspective while PROJECTION is active.
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, w, 0, h)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(w, 0)
    glTexCoord2f(1, 1); glVertex2f(w, h)
    glTexCoord2f(0, 1); glVertex2f(0, h)
    glEnd()
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDeleteTextures([tex_id])

def predict_drawing():
    img = cv2.resize(draw_canvas, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img_tensor = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return symbols_list[predicted.item()]

def main():
    global prev_x, prev_y, prev_hand_x, draw_canvas, mode, current_label, smooth_rot

    pygame.init()
    width, height = 1280, 720
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Air-Draw 3D Inspector")

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.55, 0.65, 0.72, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48.0)

    glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0.7, 0.85, 1.0, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (-5, -2, 3, 1))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.2, 0.5, 0.7, 1.0))
    glLightfv(GL_LIGHT1, GL_SPECULAR, (0.15, 0.35, 0.55, 1.0))

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    current_label = predict_drawing()
                    mode = "rotate"
                    print("Predicted:", current_label)
                elif event.key == pygame.K_c:
                    draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                    mode = "draw"
                    current_label = None

        success, live_frame = cap.read()
        if not success:
            continue
        live_frame = cv2.flip(live_frame, 1)
        rgb_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * cam_w)
            y = int(index_tip.y * cam_h)

            if mode == "draw":
                if prev_x is not None:
                    cv2.line(draw_canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                prev_x, prev_y = x, y
            else:
                if prev_hand_x is not None:
                    delta = x - prev_hand_x
                    smooth_rot += delta * 0.5
                prev_hand_x = x
                prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None
            prev_hand_x = None

        glClearColor(0.03, 0.05, 0.09, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        # The stage center maps to the visual center of the left viewport, not the window.
        glTranslatef(-1.2, 0.18, -7.0)
        # Shape rotation math: horizontal index-fingertip movement changes the angle.
        glRotatef(smooth_rot, 0, 0, 1)
        glRotatef(30, 1, 0, 0)

        if current_label and current_label in shape_functions:
            glScalef(shape_scales.get(current_label, 1.0), shape_scales.get(current_label, 1.0),
                     shape_scales.get(current_label, 1.0))
            shape_functions[current_label]()

        # HUD compositing: draw the transparent textured overlay after the 3D scene.
        hud_tex = render_hud(width, height, live_frame)
        draw_hud_fullscreen(hud_tex, width, height)

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()